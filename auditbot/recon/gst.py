"""GST reconciliations.

* purchase register  <->  GSTR-2B      (ITC eligibility / ITC at risk)
* sales register     <->  GSTR-1       (outward supply reporting)
* books liability    <->  GSTR-3B      (tax declared)
* GSTR-3B liability  <->  cash+ITC set-off (CGST/SGST/IGST payment tracking)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..matching import Pass, cascade_match
from ..normalize import r2, validate_gstin

TAX_COLS = ("taxable", "cgst", "sgst", "igst", "cess", "total")


def _passes(cfg: Config) -> list[Pass]:
    """Invoice matching cascade: the supplier's invoice number on the portal
    rarely matches the way the accountant keyed it, so we step down."""
    atol = float(cfg.th("amount_tolerance", 1.0))
    dtol = int(cfg.th("date_tolerance_days", 7))
    fz = float(cfg.th("fuzzy_party_threshold", 88))
    return [
        Pass("GSTIN + invoice no", keys=[("gstin", "gstin"), ("inv_key", "inv_key")],
             confidence=100),
        Pass("GSTIN + invoice + date", keys=[("gstin", "gstin")],
             amount=("total", "total"), amount_tol=max(atol, 2.0),
             date=("txn_date", "txn_date"), date_tol=dtol, confidence=96),
        Pass("party + invoice no", keys=[("party_key", "party_key"), ("inv_key", "inv_key")],
             confidence=94),
        Pass("party + value + date", keys=[("party_key", "party_key")],
             amount=("total", "total"), amount_tol=max(atol, 2.0),
             date=("txn_date", "txn_date"), date_tol=dtol, confidence=90),
        Pass("fuzzy party + value + date", amount=("total", "total"),
             amount_tol=max(atol, 2.0), date=("txn_date", "txn_date"), date_tol=dtol,
             fuzzy=("party_key", "party_key"), fuzzy_threshold=fz, confidence=82),
        Pass("value + date only", amount=("total", "total"), amount_tol=atol,
             date=("txn_date", "txn_date"), date_tol=max(dtol, 15), confidence=65),
    ]


def _diff_frame(matched: pd.DataFrame, lp: str, rp: str, tol: float) -> pd.DataFrame:
    """Per-invoice tax differences on matched pairs."""
    if matched.empty:
        return pd.DataFrame()
    out = matched.copy()
    for col in TAX_COLS:
        lc, rc = f"{lp}_{col}", f"{rp}_{col}"
        if lc in out.columns and rc in out.columns:
            out[f"diff_{col}"] = (out[lc].astype(float) - out[rc].astype(float)).round(2)
    diff_cols = [c for c in out.columns if c.startswith("diff_")]
    if not diff_cols:
        return pd.DataFrame()
    out["max_abs_diff"] = out[diff_cols].abs().max(axis=1).round(2)
    out["tax_diff"] = (out.get("diff_cgst", 0) + out.get("diff_sgst", 0)
                       + out.get("diff_igst", 0) + out.get("diff_cess", 0)).round(2)
    return out[out["max_abs_diff"] > tol].sort_values("max_abs_diff", ascending=False)


def _totals(df: pd.DataFrame, label: str) -> dict:
    if df is None or df.empty:
        return {"Source": label, "Invoices": 0, **{c.title(): 0.0 for c in TAX_COLS}}
    row = {"Source": label, "Invoices": len(df)}
    for c in TAX_COLS:
        row[c.title()] = r2(df[c].sum()) if c in df.columns else 0.0
    return row


def itc_recon(cfg: Config, purchase: pd.DataFrame,
              gstr2b: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if purchase is None or purchase.empty or gstr2b is None or gstr2b.empty:
        return {}
    res = cascade_match(purchase, gstr2b, _passes(cfg), "books", "gstr2b")
    tol = float(cfg.th("tax_tolerance", 5.0))
    matched, books_only, portal_only = res["matched"], res["left_only"], res["right_only"]

    books_only = books_only.copy()
    books_only["exception"] = "ITC claimed in books but invoice not in GSTR-2B"
    books_only["itc_at_risk"] = books_only["tax"].round(2)

    portal_only = portal_only.copy()
    portal_only["exception"] = "Invoice in GSTR-2B but not recorded in books"
    portal_only["itc_unclaimed"] = portal_only["tax"].round(2)

    diffs = _diff_frame(matched, "books", "gstr2b", tol)

    monthly = pd.DataFrame()
    if "month" in purchase.columns:
        b = purchase.groupby("month")[["taxable", "tax"]].sum().add_prefix("books_")
        p = gstr2b.assign(month=gstr2b["txn_date"].dt.to_period("M").astype(str)) \
                  .groupby("month")[["taxable", "tax"]].sum().add_prefix("gstr2b_")
        monthly = b.join(p, how="outer").fillna(0).round(2).reset_index()
        monthly["diff_taxable"] = (monthly["books_taxable"] - monthly["gstr2b_taxable"]).round(2)
        monthly["diff_tax"] = (monthly["books_tax"] - monthly["gstr2b_tax"]).round(2)

    summary = pd.DataFrame([
        _totals(purchase, "Purchase register (books)"),
        _totals(gstr2b, "GSTR-2B (portal)"),
    ])
    stats = pd.DataFrame([{
        "Invoices in books": len(purchase),
        "Invoices in GSTR-2B": len(gstr2b),
        "Matched": len(matched),
        "Match rate %": r2(100 * len(matched) / max(len(purchase), 1)),
        "In books, not in 2B": len(books_only),
        "ITC at risk": r2(books_only["tax"].sum()),
        "In 2B, not in books": len(portal_only),
        "ITC unclaimed": r2(portal_only["tax"].sum()),
        "Value mismatches": len(diffs),
        "Net tax difference": r2(diffs["tax_diff"].sum()) if not diffs.empty else 0.0,
    }])
    return {"itc_summary": summary, "itc_stats": stats, "itc_matched": matched,
            "itc_books_only": books_only, "itc_portal_only": portal_only,
            "itc_value_diff": diffs, "itc_monthly": monthly}


def outward_recon(cfg: Config, sales: pd.DataFrame,
                  gstr1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Books outward supplies vs GSTR-1 (summary level, month by month)."""
    if sales is None or sales.empty:
        return {}
    books = (sales.groupby("month")[["taxable", "cgst", "sgst", "igst", "cess", "tax"]]
                  .sum().round(2).add_prefix("books_").reset_index()
                  .rename(columns={"month": "period"}))
    if gstr1 is None or gstr1.empty:
        return {"sales_monthly": books}
    portal = (gstr1.groupby("period")[["taxable", "cgst", "sgst", "igst", "cess", "tax"]]
                   .sum().round(2).add_prefix("gstr1_").reset_index())
    m = books.merge(portal, on="period", how="outer").fillna(0.0)
    m["diff_taxable"] = (m["books_taxable"] - m["gstr1_taxable"]).round(2)
    m["diff_tax"] = (m["books_tax"] - m["gstr1_tax"]).round(2)
    tol = float(cfg.th("tax_tolerance", 5.0))
    m["status"] = np.where(m[["diff_taxable", "diff_tax"]].abs().max(axis=1) <= tol,
                           "AGREED", "MISMATCH")
    return {"sales_monthly": m.sort_values("period")}


def payment_tracking(cfg: Config, gstr3b: pd.DataFrame, sales: pd.DataFrame,
                     purchase: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """CGST / SGST / IGST liability, set-off and cash payment, month by month."""
    if gstr3b is None or gstr3b.empty:
        return {}
    g = gstr3b.copy()
    g["liability"] = g["tax"].round(2)
    g["itc_available"] = g["itc"].round(2)
    g["paid_total"] = (g["paid_cash"] + g["paid_credit"]).round(2)
    g["shortfall"] = (g["liability"] - g["paid_total"]).round(2)
    g["itc_utilisation %"] = np.where(
        g["liability"] > 0, (100 * g["paid_credit"] / g["liability"]).round(2), 0.0)
    g["status"] = np.where(g["shortfall"].abs() <= float(cfg.th("tax_tolerance", 5.0)),
                           "PAID IN FULL",
                           np.where(g["shortfall"] > 0, "SHORT PAID", "EXCESS PAID"))
    g["challan_missing"] = (g["paid_cash"] > 0) & (g["challan_no"].astype(str).str.len() < 3)

    cols = ["period", "taxable", "cgst", "sgst", "igst", "cess", "liability",
            "itc_available", "paid_credit", "paid_cash", "paid_total", "shortfall",
            "itc_utilisation %", "challan_no", "paid_date", "status", "challan_missing"]
    detail = g[[c for c in cols if c in g.columns]].sort_values("period")

    # books liability vs 3B declared
    comparison = pd.DataFrame()
    if sales is not None and not sales.empty:
        books = (sales.groupby("month")[["taxable", "tax"]].sum().round(2)
                      .add_prefix("books_").reset_index().rename(columns={"month": "period"}))
        comparison = books.merge(
            g[["period", "taxable", "tax"]].rename(
                columns={"taxable": "gstr3b_taxable", "tax": "gstr3b_tax"}),
            on="period", how="outer").fillna(0.0)
        comparison["diff_taxable"] = (comparison["books_taxable"]
                                      - comparison["gstr3b_taxable"]).round(2)
        comparison["diff_tax"] = (comparison["books_tax"] - comparison["gstr3b_tax"]).round(2)

    totals = pd.DataFrame([{
        "Periods filed": len(g),
        "Total liability": r2(g["liability"].sum()),
        "Paid by ITC": r2(g["paid_credit"].sum()),
        "Paid in cash": r2(g["paid_cash"].sum()),
        "Net shortfall": r2(g["shortfall"].sum()),
        "Periods short paid": int((g["status"] == "SHORT PAID").sum()),
        "Challan reference missing": int(g["challan_missing"].sum()),
    }])
    return {"gst_payment": detail, "gst_payment_totals": totals,
            "gst_books_vs_3b": comparison}


def gstin_quality(purchase: pd.DataFrame, sales: pd.DataFrame,
                  cfg: Config) -> pd.DataFrame:
    """Every invoice whose counter-party GSTIN fails format/check-digit, plus
    intra/inter-state tax-head sanity against the place of supply."""
    rows = []
    home = cfg.state_code
    for df, side in ((purchase, "Purchase"), (sales, "Sales")):
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            ok, reason = validate_gstin(r.get("gstin"))
            issues = [] if ok else [reason]
            g = str(r.get("gstin") or "")
            if ok:
                same_state = g[:2] == home
                igst, cgst = float(r.get("igst", 0)), float(r.get("cgst", 0))
                if same_state and igst > 0:
                    issues.append("IGST charged on an intra-state supply")
                if not same_state and cgst > 0:
                    issues.append("CGST/SGST charged on an inter-state supply")
            if issues:
                rows.append({"side": side, "row_id": r.get("row_id"),
                             "date": r.get("txn_date"), "party": r.get("party"),
                             "gstin": g, "invoice_no": r.get("invoice_no"),
                             "total": r.get("total"), "issue": "; ".join(issues)})
    return pd.DataFrame(rows)


def run(data: dict[str, pd.DataFrame], cfg: Config) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    out.update(itc_recon(cfg, data.get("purchase"), data.get("gstr2b")))
    out.update(outward_recon(cfg, data.get("sales"), data.get("gstr1")))
    out.update(payment_tracking(cfg, data.get("gstr3b"), data.get("sales"),
                                data.get("purchase")))
    q = gstin_quality(data.get("purchase"), data.get("sales"), cfg)
    if not q.empty:
        out["gstin_issues"] = q
    return out
