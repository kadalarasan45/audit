"""Excel working papers.

One workbook, indexed, formatted and filterable -- the file the audit senior
actually opens. Sheets are declared in :data:`LAYOUT`; anything with no data
is skipped automatically, so a trading client simply gets no inventory
sheets.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .exceptions import rule_catalogue
from .pipeline import Result

log = logging.getLogger("auditbot.excel")

INR = "#,##,##0.00"        # Indian lakh/crore grouping
DATE = "dd-mmm-yyyy"

# (sheet name, heading, source, key)  -- source: "recon" | "data" | "special"
LAYOUT: list[tuple[str, str, str, str]] = [
    ("01 Exceptions", "Audit exception register", "special", "register"),
    ("02 Exception Summary", "Exceptions by area and severity", "special", "summary"),
    ("03 BRS", "Bank reconciliation statement", "recon", "brs"),
    ("04 Bank Summary", "Bank reconciliation - summary", "recon", "bank_summary"),
    ("05 Bank Not In Books", "In the bank statement but not in the books", "recon", "bank_only"),
    ("06 Cheques Uncleared", "In the books but not yet through the bank", "recon", "book_only"),
    ("07 Bank Matched", "Matched bank transactions", "recon", "bank_matched"),
    ("08 Debtors Summary", "Debtors - summary", "recon", "debtors_summary"),
    ("09 Debtors Party", "Debtors - party-wise position", "recon", "debtors_party"),
    ("10 Debtors Ageing", "Debtors - ageing buckets", "recon", "debtors_pivot"),
    ("11 Debtors Bills", "Debtors - bill-wise ageing", "recon", "debtors_detail"),
    ("12 Creditors Summary", "Creditors - summary", "recon", "creditors_summary"),
    ("13 Creditors Party", "Creditors - party-wise position", "recon", "creditors_party"),
    ("14 Creditors Ageing", "Creditors - ageing buckets", "recon", "creditors_pivot"),
    ("15 Creditors Bills", "Creditors - bill-wise ageing", "recon", "creditors_detail"),
    ("16 Billwise Receipts", "Receipts applied bill-wise", "recon", "receipts_allocations"),
    ("17 Receipt Bill Status", "Sales bills - settlement status", "recon", "receipts_bill_status"),
    ("18 Receipts On Account", "Receipts not applied to any bill", "recon", "receipts_unapplied"),
    ("19 Collection Days", "Collection performance by party", "recon", "receipts_performance"),
    ("20 Billwise Payments", "Payments applied bill-wise", "recon", "payments_allocations"),
    ("21 Payment Bill Status", "Purchase bills - settlement status", "recon", "payments_bill_status"),
    ("22 Payments On Account", "Payments not applied to any bill", "recon", "payments_unapplied"),
    ("23 Payment Days", "Payment performance by party", "recon", "payments_performance"),
    ("24 ITC Summary", "Purchase register vs GSTR-2B - totals", "recon", "itc_summary"),
    ("25 ITC Stats", "Purchase register vs GSTR-2B - match statistics", "recon", "itc_stats"),
    ("26 ITC Not In 2B", "ITC claimed but invoice not in GSTR-2B", "recon", "itc_books_only"),
    ("27 2B Not In Books", "In GSTR-2B but not recorded in the books", "recon", "itc_portal_only"),
    ("28 ITC Value Diff", "Invoice value differences vs GSTR-2B", "recon", "itc_value_diff"),
    ("29 ITC Monthly", "ITC - month-wise comparison", "recon", "itc_monthly"),
    ("30 GST Payment", "CGST / SGST / IGST liability and payment", "recon", "gst_payment"),
    ("31 GST Payment Totals", "GST payment - totals", "recon", "gst_payment_totals"),
    ("32 Books vs GSTR-3B", "Books outward supply vs GSTR-3B", "recon", "gst_books_vs_3b"),
    ("33 Books vs GSTR-1", "Books outward supply vs GSTR-1", "recon", "sales_monthly"),
    ("34 GSTIN Issues", "GSTIN validity and tax-head issues", "recon", "gstin_issues"),
    ("35 Cash Above Limit", "Cash transactions above the statutory limit", "special", "cash"),
    ("36 Sales Control", "Sales - party-wise control account", "recon", "sales_control"),
    ("37 Sales Trend", "Sales - month-wise trend", "recon", "sales_trend"),
    ("38 Purchase Control", "Purchases - party-wise control account", "recon", "purchase_control"),
    ("39 Purchase Trend", "Purchases - month-wise trend", "recon", "purchase_trend"),
    ("40 Invoice Sequence", "Sales invoice series - gaps and repeats", "recon", "invoice_sequence"),
    ("41 Salary Summary", "Salary - summary", "recon", "salary_summary"),
    ("42 Salary Monthly", "Salary - month-wise cost", "recon", "salary_monthly"),
    ("43 Salary Employee", "Salary - employee-wise", "recon", "salary_employee"),
    ("44 Salary Detail", "Salary - full register with statutory checks", "recon", "salary_detail"),
    ("45 Salary Bank Tieup", "Payroll net pay vs bank salary debits", "recon", "salary_bank_tieup"),
    ("46 Inventory Summary", "Inventory - summary", "recon", "inventory_summary"),
    ("47 Inventory Detail", "Inventory - item-wise movement", "recon", "inventory_detail"),
    ("48 Inventory Group", "Inventory - group-wise value", "recon", "inventory_group"),
    ("49 Conversion Summary", "Conversion - summary", "recon", "conversion_summary"),
    ("50 Conversion Detail", "Conversion - voucher-wise yield", "recon", "conversion_detail"),
    ("51 Conversion Item", "Conversion - item-wise yield", "recon", "conversion_item"),
    ("52 Trade Summary", "Purchase / sales control totals", "recon", "trade_summary"),
    ("97 Rule Catalogue", "Checks performed by the system", "special", "rules"),
    ("98 Run Log", "Processing log", "special", "runlog"),
    ("99 Data Loaded", "Source records loaded", "special", "sources"),
]


def _cash_sheet(res: Result) -> pd.DataFrame:
    """Every cash transaction at or above the s.40A(3) / 269ST thresholds."""
    cfg = res.cfg
    pay_limit = float(cfg.th("cash_payment_limit", 10000))
    rec_limit = float(cfg.th("cash_receipt_limit", 200000))
    rows = []
    pay = res.data.get("payments")
    if pay is not None and not pay.empty:
        hits = pay[pay["is_cash"] & (pay["amount"] > pay_limit)]
        for _, r in hits.iterrows():
            rows.append({"Type": "Cash payment", "Date": r["txn_date"], "Party": r["party"],
                         "Voucher": r["voucher_no"], "Amount": r["amount"],
                         "Limit": pay_limit, "Excess": round(r["amount"] - pay_limit, 2),
                         "Provision": "s.40A(3) - disallowance of expenditure",
                         "Narration": r.get("narration", "")})
    rec = res.data.get("receipts")
    if rec is not None and not rec.empty:
        hits = rec[rec["is_cash"] & (rec["amount"] >= rec_limit)]
        for _, r in hits.iterrows():
            rows.append({"Type": "Cash receipt", "Date": r["txn_date"], "Party": r["party"],
                         "Voucher": r["voucher_no"], "Amount": r["amount"],
                         "Limit": rec_limit, "Excess": round(r["amount"] - rec_limit, 2),
                         "Provision": "s.269ST - penalty u/s 271DA",
                         "Narration": r.get("narration", "")})
    cash = res.data.get("cashbook")
    if cash is not None and not cash.empty:
        hits = cash[cash["amount"].abs() > pay_limit]
        for _, r in hits.iterrows():
            rows.append({"Type": "Cash book " + ("payment" if r["amount"] < 0 else "receipt"),
                         "Date": r["txn_date"], "Party": r["description"], "Voucher": r["ref_no"],
                         "Amount": abs(r["amount"]), "Limit": pay_limit,
                         "Excess": round(abs(r["amount"]) - pay_limit, 2),
                         "Provision": "s.40A(3) / s.269ST - verify",
                         "Narration": r["description"]})
    df = pd.DataFrame(rows)
    return df.sort_values("Amount", ascending=False) if not df.empty else df


def _source_sheet(res: Result) -> pd.DataFrame:
    rows = []
    for name, df in res.data.items():
        if df is None or df.empty:
            rows.append({"Source": name, "Records": 0, "Files": "", "Period from": None,
                         "Period to": None})
            continue
        files = ", ".join(sorted(set(df.get("source_file", pd.Series(dtype=str)).dropna())))
        dates = df["txn_date"] if "txn_date" in df.columns else \
            (df["bill_date"] if "bill_date" in df.columns else pd.Series(dtype="datetime64[ns]"))
        rows.append({"Source": name, "Records": len(df), "Files": files,
                     "Period from": dates.min() if len(dates) else None,
                     "Period to": dates.max() if len(dates) else None})
    return pd.DataFrame(rows)


def _kpi_sheet(res: Result) -> pd.DataFrame:
    labels = {
        "sales": "Sales (register)", "purchases": "Purchases (register)",
        "receipts": "Receipts", "payments": "Payments",
        "debtors": "Debtors closing", "creditors": "Creditors closing",
        "salary_cost": "Salary cost (gross)", "closing_stock": "Closing stock",
        "bank_lines": "Bank statement lines", "bank_match_rate": "Bank match rate %",
        "bank_unreconciled": "Bank unreconciled difference",
        "bank_only": "Bank entries not in books", "book_only": "Cheques uncleared",
        "gst_match_rate": "GSTR-2B match rate %", "itc_at_risk": "ITC at risk",
        "itc_unclaimed": "ITC unclaimed", "gst_shortfall": "GST short paid",
        "debtors_overdue": "Debtors overdue", "creditors_overdue": "Creditors overdue",
        "unapplied_receipts": "Receipts on account", "unapplied_payments": "Payments on account",
        "cash_above_limit": "Cash exceptions", "duplicates": "Duplicate exceptions",
        "exceptions_total": "Total exceptions", "exceptions_high": "High severity",
        "exceptions_medium": "Medium severity", "exceptions_low": "Low severity",
        "exception_value": "Value involved in exceptions", "rules_run": "Checks performed",
    }
    return pd.DataFrame([{"Indicator": v, "Value": res.kpis.get(k, 0)}
                         for k, v in labels.items()])


def _frame_for(res: Result, source: str, key: str) -> pd.DataFrame:
    if source == "recon":
        return res.recon.get(key, pd.DataFrame())
    if source == "data":
        return res.data.get(key, pd.DataFrame())
    return {
        "register": res.register,
        "summary": res.summary,
        "cash": _cash_sheet(res),
        "rules": rule_catalogue(),
        "runlog": pd.DataFrame(res.log),
        "sources": _source_sheet(res),
    }.get(key, pd.DataFrame())


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop machine-only columns and make datetimes timezone-naive for Excel."""
    drop = [c for c in df.columns
            if c.startswith("x_") or c in {"dataset", "source_sheet", "party_key",
                                           "inv_key", "bill_key", "item_key",
                                           "emp_key", "output_key", "input_key",
                                           "left_idx", "right_idx", "_open", "_left"}]
    out = df.drop(columns=drop, errors="ignore").copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.DatetimeTZDtype):
            out[c] = out[c].dt.tz_localize(None)
        elif out[c].dtype == "object":
            out[c] = out[c].map(lambda v: str(v) if isinstance(v, (list, dict, set)) else v)
    return out


def _pretty(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def write(res: Result, path: Path | None = None) -> Path:
    cfg = res.cfg
    path = Path(path or (res.outdir / str(cfg.get("report.workbook_name",
                                                  "Audit_Working_Papers.xlsx"))))
    path.parent.mkdir(parents=True, exist_ok=True)

    sheets: list[tuple[str, str, pd.DataFrame]] = []
    for name, heading, source, key in LAYOUT:
        df = _frame_for(res, source, key)
        if df is None or df.empty:
            continue
        sheets.append((name[:31], heading, _clean(df)))

    with pd.ExcelWriter(path, engine="xlsxwriter",
                        datetime_format=DATE, date_format=DATE) as xl:
        wb = xl.book
        fmt = {
            "title": wb.add_format({"bold": True, "font_size": 15, "font_color": "#0F2B46"}),
            "sub": wb.add_format({"font_size": 9, "font_color": "#5A6B7B", "italic": True}),
            "head": wb.add_format({"bold": True, "bg_color": "#0F2B46", "font_color": "white",
                                   "border": 1, "text_wrap": True, "valign": "vcenter"}),
            "num": wb.add_format({"num_format": INR}),
            "date": wb.add_format({"num_format": DATE}),
            "text": wb.add_format({"text_wrap": False}),
            "index": wb.add_format({"font_color": "#1F5FA8", "underline": 1}),
            "kpi_l": wb.add_format({"bold": True, "bg_color": "#F2F6FA", "border": 1}),
            "kpi_v": wb.add_format({"num_format": INR, "bg_color": "#F2F6FA", "border": 1,
                                    "bold": True}),
            "high": wb.add_format({"bg_color": "#FBE3E4", "font_color": "#9C0006"}),
            "med": wb.add_format({"bg_color": "#FFF3CD", "font_color": "#8A6100"}),
            "low": wb.add_format({"bg_color": "#EAF3EA", "font_color": "#36613A"}),
        }

        # ---------------------------------------------------- index sheet
        idx = wb.add_worksheet("00 Index")
        idx.hide_gridlines(2)
        idx.set_column("A:A", 4)
        idx.set_column("B:B", 30)
        idx.set_column("C:C", 58)
        idx.set_column("D:D", 12)
        idx.write("B2", f"{cfg.client}", fmt["title"])
        idx.write("B3", f"Audit working papers - FY {cfg.fy}  "
                        f"({cfg.period_from.date()} to {cfg.period_to.date()})", fmt["sub"])
        idx.write("B4", f"Generated {res.kpis.get('generated', '')} by auditbot  |  "
                        f"{res.kpis.get('rules_run', 0)} checks  |  "
                        f"{len(res.register)} exceptions "
                        f"({res.kpis.get('exceptions_high', 0)} high)", fmt["sub"])

        kpi = _kpi_sheet(res)
        row = 6
        idx.write("B6", "Key indicators", fmt["head"])
        idx.write("C6", "", fmt["head"])
        for _, r in kpi.iterrows():
            row += 1
            idx.write(row, 1, r["Indicator"], fmt["kpi_l"])
            idx.write(row, 2, r["Value"], fmt["kpi_v"])

        row += 3
        idx.write(row, 1, "Sheet", fmt["head"])
        idx.write(row, 2, "Contents", fmt["head"])
        idx.write(row, 3, "Rows", fmt["head"])
        for name, heading, df in sheets:
            row += 1
            idx.write_url(row, 1, f"internal:'{name}'!A1", fmt["index"], name)
            idx.write(row, 2, heading)
            idx.write(row, 3, len(df))
        idx.freeze_panes(6, 0)

        # ------------------------------------------------------- data sheets
        for name, heading, df in sheets:
            df.to_excel(xl, sheet_name=name, startrow=2, index=False)
            ws = xl.sheets[name]
            ws.hide_gridlines(2)
            ws.write("A1", heading, fmt["title"])
            ws.write_url("A2", "internal:'00 Index'!A1", fmt["index"], "< back to index")

            for col, header in enumerate(df.columns):
                ws.write(2, col, _pretty(str(header)), fmt["head"])
                series = df[header]
                width = max(len(str(header)) + 3,
                            int(series.astype(str).str.len().quantile(0.92)) + 2
                            if len(series) else 12)
                width = min(max(width, 10), 55)
                if pd.api.types.is_datetime64_any_dtype(series):
                    ws.set_column(col, col, max(width, 13), fmt["date"])
                elif pd.api.types.is_numeric_dtype(series) and not \
                        pd.api.types.is_bool_dtype(series):
                    ws.set_column(col, col, max(width, 14), fmt["num"])
                else:
                    ws.set_column(col, col, width, fmt["text"])

            last_row, last_col = len(df) + 2, len(df.columns) - 1
            if last_col >= 0 and len(df):
                ws.autofilter(2, 0, last_row, last_col)
            ws.freeze_panes(3, 0)

            if "severity" in df.columns:
                c = df.columns.get_loc("severity")
                rng = (3, c, last_row, c)
                for value, style in (("HIGH", "high"), ("MEDIUM", "med"), ("LOW", "low")):
                    ws.conditional_format(*rng, {"type": "cell", "criteria": "==",
                                                 "value": f'"{value}"', "format": fmt[style]})

    log.info("workbook written: %s (%d sheets)", path, len(sheets) + 1)
    return path
