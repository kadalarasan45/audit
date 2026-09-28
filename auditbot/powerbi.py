"""Flat CSV exports shaped for Power BI / Excel Power Query.

A star schema: one row per fact, keys instead of nested frames, and a date
dimension so time intelligence works without any modelling in the report.
Point Power BI at the folder ("Get Data > Folder") and every refresh picks up
the latest run.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .normalize import norm_party
from .pipeline import Result

log = logging.getLogger("auditbot.powerbi")


def _dim_date(cfg) -> pd.DataFrame:
    days = pd.date_range(cfg.period_from, cfg.period_to, freq="D")
    d = pd.DataFrame({"date": days})
    d["year"] = d["date"].dt.year
    d["month_no"] = d["date"].dt.month
    d["month"] = d["date"].dt.strftime("%Y-%m")
    d["month_name"] = d["date"].dt.strftime("%b %Y")
    d["quarter"] = "Q" + d["date"].dt.quarter.astype(str)
    # Indian financial year: April to March
    d["fy"] = (d["date"] - pd.DateOffset(months=3)).dt.year.astype(str) + "-" + \
              ((d["date"] - pd.DateOffset(months=3)).dt.year + 1).astype(str).str[-2:]
    d["fy_month_no"] = ((d["month_no"] - 4) % 12) + 1
    d["day_name"] = d["date"].dt.day_name()
    d["is_weekend"] = d["date"].dt.dayofweek >= 5
    d["is_holiday"] = d["date"].isin(cfg.holidays)
    return d


def _fact_transactions(res: Result) -> pd.DataFrame:
    """Every money movement in one long table, so a single visual can slice
    sales/purchases/receipts/payments/bank together."""
    specs = [
        ("sales", "Sales", "party", "total", "txn_date"),
        ("purchase", "Purchase", "party", "total", "txn_date"),
        ("receipts", "Receipt", "party", "amount", "txn_date"),
        ("payments", "Payment", "party", "amount", "txn_date"),
        ("bank", "Bank", "description", "amount", "txn_date"),
        ("cashbook", "Cash", "description", "amount", "txn_date"),
    ]
    parts = []
    for key, label, party_col, amt_col, date_col in specs:
        df = res.data.get(key)
        if df is None or df.empty or amt_col not in df.columns:
            continue
        parts.append(pd.DataFrame({
            "row_id": df["row_id"],
            "txn_type": label,
            "date": df[date_col],
            "party": df[party_col].astype(str),
            "party_key": df.get("party_key", df[party_col].map(norm_party)),
            "amount": df[amt_col].astype(float).round(2),
            "abs_amount": df[amt_col].astype(float).abs().round(2),
            "direction": "IN" if label in ("Sales", "Receipt") else "OUT",
            "mode": df.get("mode", pd.Series([""] * len(df))).astype(str),
            "is_cash": df.get("is_cash", pd.Series([False] * len(df))),
            "category": df.get("category", pd.Series([label] * len(df))).astype(str),
            "reference": df.get("invoice_no", df.get("voucher_no",
                                                     pd.Series([""] * len(df)))).astype(str),
            "source_file": df.get("source_file", pd.Series([""] * len(df))).astype(str),
        }))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["month"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m")
    return out


def _dim_party(res: Result) -> pd.DataFrame:
    rows = []
    for key, role in (("sales", "Customer"), ("purchase", "Supplier")):
        df = res.data.get(key)
        if df is None or df.empty:
            continue
        g = (df.groupby(["party_key", "party"])
               .agg(gstin=("gstin", "first"), invoices=("total", "size"),
                    turnover=("total", "sum"), first_txn=("txn_date", "min"),
                    last_txn=("txn_date", "max")).reset_index())
        g["role"] = role
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["turnover"] = out["turnover"].round(2)

    for key, col in (("debtors", "closing_receivable"), ("creditors", "closing_payable")):
        df = res.data.get(key)
        if df is None or df.empty:
            out[col] = 0.0
            continue
        bal = df.groupby("party_key")["balance"].sum().round(2)
        out[col] = out["party_key"].map(bal).fillna(0.0)
    return out


def _fact_ageing(res: Result) -> pd.DataFrame:
    parts = []
    for kind in ("debtors", "creditors"):
        d = res.recon.get(f"{kind}_detail")
        if d is None or d.empty:
            continue
        parts.append(pd.DataFrame({
            "side": kind.title(), "party": d["party"], "party_key": d["party_key"],
            "bill_no": d["bill_no"], "bill_date": d["bill_date"], "due_date": d["due_date"],
            "bill_amount": d["bill_amount"].round(2), "balance": d["balance"].round(2),
            "age_days": d["age_days"], "overdue_days": d["overdue_days"],
            "bucket": d["bucket"], "overdue": d["overdue"],
        }))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _fact_gst(res: Result) -> pd.DataFrame:
    g = res.recon.get("gst_payment")
    if g is None or g.empty:
        return pd.DataFrame()
    out = g.copy()
    out["period_start"] = pd.PeriodIndex(out["period"], freq="M").to_timestamp()
    return out


def _fact_bank(res: Result) -> pd.DataFrame:
    parts = []
    for key, status in (("bank_matched", "Matched"), ("bank_only", "In bank only"),
                        ("book_only", "In books only")):
        d = res.recon.get(key)
        if d is None or d.empty:
            continue
        if key == "bank_matched":
            sub = pd.DataFrame({"date": d["bank_txn_date"], "description": d["bank_description"],
                                "amount": d["bank_amount"], "category": d["bank_category"],
                                "match_pass": d["match_pass"], "confidence": d["confidence"]})
        else:
            sub = pd.DataFrame({"date": d["txn_date"], "description": d["description"],
                                "amount": d["amount"], "category": d["category"],
                                "match_pass": "", "confidence": 0.0})
            sub["reason"] = d.get("reason", "")
        sub["status"] = status
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["month"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m")
    return out


def _kpi_table(res: Result) -> pd.DataFrame:
    return pd.DataFrame([{"measure": k, "value": v} for k, v in res.kpis.items()])


EXPORTS = {
    "dim_date": lambda r: _dim_date(r.cfg),
    "dim_party": _dim_party,
    "fact_transactions": _fact_transactions,
    "fact_exceptions": lambda r: r.register,
    "fact_exception_summary": lambda r: r.summary,
    "fact_ageing": _fact_ageing,
    "fact_gst_payment": _fact_gst,
    "fact_bank_recon": _fact_bank,
    "fact_itc_books_only": lambda r: r.recon.get("itc_books_only", pd.DataFrame()),
    "fact_itc_portal_only": lambda r: r.recon.get("itc_portal_only", pd.DataFrame()),
    "fact_salary_monthly": lambda r: r.recon.get("salary_monthly", pd.DataFrame()),
    "fact_salary_employee": lambda r: r.recon.get("salary_employee", pd.DataFrame()),
    "fact_inventory": lambda r: r.recon.get("inventory_detail", pd.DataFrame()),
    "fact_conversion": lambda r: r.recon.get("conversion_detail", pd.DataFrame()),
    "fact_sales_trend": lambda r: r.recon.get("sales_trend", pd.DataFrame()),
    "fact_purchase_trend": lambda r: r.recon.get("purchase_trend", pd.DataFrame()),
    "fact_billwise_receipts": lambda r: r.recon.get("receipts_allocations", pd.DataFrame()),
    "fact_billwise_payments": lambda r: r.recon.get("payments_allocations", pd.DataFrame()),
    "kpi": _kpi_table,
}

README = """# Power BI / Power Query data model

Every file here is one refresh of `auditbot`. To build the report:

1. Power BI Desktop > Get Data > Folder > point at this folder > Combine & Load,
   or Get Data > Text/CSV for individual tables.
2. Mark `dim_date[date]` as the date table.
3. Create the relationships:
   - dim_date[date]            1 --> * fact_transactions[date]
   - dim_date[date]            1 --> * fact_bank_recon[date]
   - dim_date[date]            1 --> * fact_ageing[bill_date]
   - dim_date[date]            1 --> * fact_exceptions[date]
   - dim_party[party_key]      1 --> * fact_transactions[party_key]
   - dim_party[party_key]      1 --> * fact_ageing[party_key]

Suggested measures:

    Exceptions High   = CALCULATE(COUNTROWS(fact_exceptions),
                                  fact_exceptions[severity] = "HIGH")
    Exception Value   = SUM(fact_exceptions[amount])
    ITC at Risk       = SUM(fact_itc_books_only[tax])
    Overdue Debtors   = CALCULATE(SUM(fact_ageing[balance]),
                                  fact_ageing[side] = "Debtors",
                                  fact_ageing[overdue] = TRUE())
    Collection Days   = AVERAGE(fact_billwise_receipts[days_taken])
    GST Shortfall     = SUM(fact_gst_payment[shortfall])

`kpi.csv` holds the same headline numbers the HTML dashboard shows, so a card
visual can read them directly without any DAX.
"""


def write(res: Result, folder: Path | None = None) -> Path:
    folder = Path(folder or (res.outdir / "powerbi"))
    folder.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, fn in EXPORTS.items():
        try:
            df = fn(res)
        except Exception as exc:
            log.error("export %s failed: %s", name, exc)
            continue
        if df is None or df.empty:
            continue
        df.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")
        written += 1
    (folder / "README.md").write_text(README, encoding="utf-8")
    log.info("power bi exports: %d tables in %s", written, folder)
    return folder
