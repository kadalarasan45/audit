"""Turn mapped frames into typed, classified canonical datasets."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import ROOT, Config
from .io_utils import load_frames
from .normalize import (clean_amount, clean_gstin, display_party, norm_invoice,
                        norm_party, parse_date, parse_period, r2)

log = logging.getLogger("auditbot.ingest")


def _col(df: pd.DataFrame, name: str, default=None) -> pd.Series:
    if name in df.columns:
        s = df[name]
        if default == "":
            # stringifying a NaN gives the literal "nan", which then leaks into
            # every report and every match key
            return s.where(s.notna(), "").astype(object)
        return s
    return pd.Series([default] * len(df), index=df.index)


def _amt(df: pd.DataFrame, name: str) -> pd.Series:
    return _col(df, name, 0).map(clean_amount).astype(float)


def _date(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_datetime(_col(df, name).map(parse_date), errors="coerce")


def _tag(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Stable row id used by every exception back to its source row."""
    df = df.reset_index(drop=True)
    df.insert(0, "row_id", [f"{kind[:3].upper()}{i + 1:05d}" for i in range(len(df))])
    df["dataset"] = kind
    return df


def _keep_extras(src: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    for c in ("source_file", "source_sheet"):
        if c in src.columns:
            out[c] = src[c].values
    return out


# ------------------------------------------------------------------ bank

def classify_bank(description: str, cfg: Config) -> str:
    """Bucket a bank narration. First keyword family to match wins, and the
    order below is deliberate -- 'GST on charges' is a charge, not a tax."""
    d = str(description or "").lower()
    order = [
        ("CONTRA", "contra_keywords"),
        ("CHARGES", "charges_keywords"),
        ("SALARY", "salary_keywords"),
        ("STATUTORY", "statutory_keywords"),
        ("TAX", "tax_keywords"),
        ("INTEREST", "interest_keywords"),
        ("LOAN", "loan_keywords"),
        ("CASH", "cash_keywords"),
    ]
    for label, key in order:
        if any(k in d for k in cfg.keywords(key)):
            return label
    return "TRADE"


def load_bank(cfg: Config) -> pd.DataFrame:
    raw = load_frames(cfg.source_patterns("bank"), "bank", cfg, ROOT)
    if raw.empty:
        return pd.DataFrame()

    debit, credit = _amt(raw, "debit").abs(), _amt(raw, "credit").abs()
    amount = _amt(raw, "amount")
    if "amount" in raw.columns and (debit.sum() + credit.sum()) == 0:
        # single signed/unsigned amount column plus an optional Dr/Cr flag
        drcr = _col(raw, "drcr", "").astype(str).str.strip().str.upper().str[:2]
        signed = amount.where(~drcr.isin(["DR", "D", "W"]), -amount.abs())
        signed = signed.where(~drcr.isin(["CR", "C"]), amount.abs())
        debit = signed.clip(upper=0).abs()
        credit = signed.clip(lower=0)

    out = pd.DataFrame({
        "txn_date": _date(raw, "txn_date"),
        "value_date": _date(raw, "value_date"),
        "description": _col(raw, "description", "").astype(str).str.strip(),
        "ref_no": _col(raw, "ref_no", "").astype(str).str.strip(),
        "debit": debit.round(2),
        "credit": credit.round(2),
        "balance": _amt(raw, "balance").round(2),
    })
    out["amount"] = (out["credit"] - out["debit"]).round(2)   # + = money in
    out["direction"] = pd.Series(["INFLOW"] * len(out)).where(out["amount"] >= 0, "OUTFLOW")
    out["category"] = out["description"].map(lambda d: classify_bank(d, cfg))
    out["is_cash"] = out["category"].eq("CASH")
    out["party_key"] = out["description"].map(norm_party)
    out = _keep_extras(raw, out)
    out = out[out["txn_date"].notna() & (out["amount"] != 0)]
    return _tag(out, "bank")


def _load_book_ledger(cfg: Config, source: str, kind: str) -> pd.DataFrame:
    """Books side of a cash/bank ledger (Tally export).

    Sign convention is the mirror image of a bank statement: in the books a
    *debit* to the bank ledger is money coming in.
    """
    raw = load_frames(cfg.source_patterns(source), "bank", cfg, ROOT)
    if raw.empty:
        return pd.DataFrame()
    debit, credit = _amt(raw, "debit").abs(), _amt(raw, "credit").abs()
    out = pd.DataFrame({
        "txn_date": _date(raw, "txn_date"),
        "description": _col(raw, "description", "").astype(str).str.strip(),
        "ref_no": _col(raw, "ref_no", "").astype(str).str.strip(),
        "debit": debit.round(2),
        "credit": credit.round(2),
        "balance": _amt(raw, "balance").round(2),
    })
    out["amount"] = (out["debit"] - out["credit"]).round(2)   # + = money in
    out["direction"] = pd.Series(["INFLOW"] * len(out)).where(out["amount"] >= 0, "OUTFLOW")
    out["category"] = out["description"].map(lambda d: classify_bank(d, cfg))
    out["is_cash"] = out["category"].eq("CASH") | (kind == "cashbook")
    out["party_key"] = out["description"].map(norm_party)
    # a Tally ledger export opens with a brought-forward line; it belongs in
    # the closing balance but must never be offered to the matcher
    out["is_opening"] = out["description"].str.contains(
        r"opening\s*balance|balance\s*b/?f|brought\s*forward|b/?f", case=False,
        regex=True, na=False)
    out = _keep_extras(raw, out)
    out = out[out["txn_date"].notna() & (out["amount"] != 0)]
    return _tag(out, kind)


def load_bankbook(cfg: Config) -> pd.DataFrame:
    return _load_book_ledger(cfg, "bankbook", "bankbook")


def load_cashbook(cfg: Config) -> pd.DataFrame:
    return _load_book_ledger(cfg, "cashbook", "cashbook")


# ------------------------------------------------------- invoice registers

def _load_invoices(cfg: Config, source: str, section: str, kind: str) -> pd.DataFrame:
    raw = load_frames(cfg.source_patterns(source), section, cfg, ROOT)
    if raw.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "txn_date": _date(raw, "txn_date"),
        "invoice_no": _col(raw, "invoice_no", "").astype(str).str.strip(),
        "party": _col(raw, "party", "").map(display_party),
        "gstin": _col(raw, "gstin", "").map(clean_gstin),
        "taxable": _amt(raw, "taxable").round(2),
        "cgst": _amt(raw, "cgst").round(2),
        "sgst": _amt(raw, "sgst").round(2),
        "igst": _amt(raw, "igst").round(2),
        "cess": _amt(raw, "cess").round(2),
        "total": _amt(raw, "total").round(2),
        "voucher_no": _col(raw, "voucher_no", "").astype(str).str.strip(),
        "hsn": _col(raw, "hsn", "").astype(str).str.strip(),
        "place": _col(raw, "place", "").astype(str).str.strip(),
        "itc_eligible": _col(raw, "itc_eligible", "").astype(str).str.strip(),
        "period_filed": _col(raw, "period", None).map(parse_period),
        "filing_date": _date(raw, "filing_date"),
    })
    out["tax"] = (out["cgst"] + out["sgst"] + out["igst"] + out["cess"]).round(2)
    # some registers give taxable but no total (or the other way round)
    out.loc[out["total"] == 0, "total"] = (out["taxable"] + out["tax"]).round(2)
    out.loc[out["taxable"] == 0, "taxable"] = (out["total"] - out["tax"]).round(2)
    out["party_key"] = out["party"].map(norm_party)
    out["inv_key"] = out["invoice_no"].map(norm_invoice)
    out["month"] = out["txn_date"].dt.to_period("M").astype(str)
    out = _keep_extras(raw, out)
    out = out[out["txn_date"].notna() | (out["total"] != 0)]
    return _tag(out, kind)


def load_purchase(cfg: Config) -> pd.DataFrame:
    return _load_invoices(cfg, "purchase", "purchase", "purchase")


def load_sales(cfg: Config) -> pd.DataFrame:
    return _load_invoices(cfg, "sales", "sales", "sales")


def load_gstr2b(cfg: Config) -> pd.DataFrame:
    return _load_invoices(cfg, "gstr2b", "gstr2b", "gstr2b")


# ------------------------------------------------------ receipts/payments

def _load_vouchers(cfg: Config, source: str, section: str, kind: str) -> pd.DataFrame:
    raw = load_frames(cfg.source_patterns(source), section, cfg, ROOT)
    if raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "txn_date": _date(raw, "txn_date"),
        "party": _col(raw, "party", "").map(display_party),
        "amount": _amt(raw, "amount").abs().round(2),
        "mode": _col(raw, "mode", "").astype(str).str.strip().str.upper(),
        "ref_no": _col(raw, "ref_no", "").astype(str).str.strip(),
        "bill_no": _col(raw, "bill_no", "").astype(str).str.strip(),
        "voucher_no": _col(raw, "voucher_no", "").astype(str).str.strip(),
        "narration": _col(raw, "narration", "").astype(str).str.strip(),
    })
    out["party_key"] = out["party"].map(norm_party)
    out["bill_key"] = out["bill_no"].map(norm_invoice)
    out["is_cash"] = out["mode"].str.contains("CASH", na=False)
    out["month"] = out["txn_date"].dt.to_period("M").astype(str)
    out = _keep_extras(raw, out)
    out = out[out["amount"] > 0]
    return _tag(out, kind)


def load_receipts(cfg: Config) -> pd.DataFrame:
    return _load_vouchers(cfg, "receipts", "receipts", "receipts")


def load_payments(cfg: Config) -> pd.DataFrame:
    return _load_vouchers(cfg, "payments", "payments", "payments")


# ------------------------------------------------------ outstanding bills

def _load_outstanding(cfg: Config, source: str, kind: str,
                      default_days: int) -> pd.DataFrame:
    raw = load_frames(cfg.source_patterns(source), "outstanding", cfg, ROOT)
    if raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "party": _col(raw, "party", "").map(display_party),
        "bill_no": _col(raw, "bill_no", "").astype(str).str.strip(),
        "bill_date": _date(raw, "bill_date"),
        "bill_amount": _amt(raw, "bill_amount").round(2),
        "balance": _amt(raw, "balance").round(2),
        "due_date": _date(raw, "due_date"),
    })
    days = _col(raw, "credit_days", None).map(
        lambda v: clean_amount(v) if str(v).strip() not in ("", "None", "nan") else None)
    out["credit_days"] = days.fillna(default_days).astype(float)
    out.loc[out["due_date"].isna(), "due_date"] = (
        out["bill_date"] + pd.to_timedelta(out["credit_days"], unit="D"))
    out.loc[out["bill_amount"] == 0, "bill_amount"] = out["balance"]
    out["party_key"] = out["party"].map(norm_party)
    out["bill_key"] = out["bill_no"].map(norm_invoice)
    out = _keep_extras(raw, out)
    out = out[(out["balance"].abs() > 0.004) | (out["bill_amount"].abs() > 0.004)]
    return _tag(out, kind)


def load_debtors(cfg: Config) -> pd.DataFrame:
    return _load_outstanding(cfg, "debtors", "debtors",
                             int(cfg.th("debtor_credit_days_default", 45)))


def load_creditors(cfg: Config) -> pd.DataFrame:
    return _load_outstanding(cfg, "creditors", "creditors",
                             int(cfg.th("creditor_credit_days_default", 45)))


# ---------------------------------------------------------------- salary

def load_salary(cfg: Config) -> pd.DataFrame:
    raw = load_frames(cfg.source_patterns("salary"), "salary", cfg, ROOT)
    if raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "month": _col(raw, "month", "").map(parse_period),
        "emp_id": _col(raw, "emp_id", "").astype(str).str.strip(),
        "emp_name": _col(raw, "emp_name", "").map(display_party),
        "designation": _col(raw, "designation", "").astype(str).str.strip(),
        "doj": _date(raw, "doj"),
        "dol": _date(raw, "dol"),
        "days_paid": _amt(raw, "days_paid"),
        "basic": _amt(raw, "basic").round(2),
        "hra": _amt(raw, "hra").round(2),
        "other_allow": _amt(raw, "other_allow").round(2),
        "gross": _amt(raw, "gross").round(2),
        "pf_employee": _amt(raw, "pf_employee").abs().round(2),
        "esi_employee": _amt(raw, "esi_employee").abs().round(2),
        "pt": _amt(raw, "pt").abs().round(2),
        "tds": _amt(raw, "tds").abs().round(2),
        "advance": _amt(raw, "advance").abs().round(2),
        "other_deduct": _amt(raw, "other_deduct").abs().round(2),
        "net": _amt(raw, "net").round(2),
        "mode": _col(raw, "mode", "").astype(str).str.strip().str.upper(),
        "bank_acct": _col(raw, "bank_acct", "").astype(str).str.strip(),
        "pan": _col(raw, "pan", "").astype(str).str.strip().str.upper(),
        "uan": _col(raw, "uan", "").astype(str).str.strip(),
    })
    out.loc[out["gross"] == 0, "gross"] = (
        out["basic"] + out["hra"] + out["other_allow"]).round(2)
    out["total_deduct"] = (out["pf_employee"] + out["esi_employee"] + out["pt"]
                           + out["tds"] + out["advance"] + out["other_deduct"]).round(2)
    out["computed_net"] = (out["gross"] - out["total_deduct"]).round(2)
    out.loc[out["net"] == 0, "net"] = out["computed_net"]
    out["is_cash"] = out["mode"].str.contains("CASH", na=False)
    out["emp_key"] = out["emp_id"].where(out["emp_id"].str.len() > 0,
                                         out["emp_name"].map(norm_party))
    out = _keep_extras(raw, out)
    out = out[(out["gross"] != 0) | (out["net"] != 0)]
    return _tag(out, "salary")


# ------------------------------------------------------------- inventory

def load_inventory(cfg: Config) -> pd.DataFrame:
    raw = load_frames(cfg.source_patterns("inventory"), "inventory", cfg, ROOT)
    if raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "item": _col(raw, "item", "").map(display_party),
        "uom": _col(raw, "uom", "").astype(str).str.strip(),
        "group": _col(raw, "group", "").astype(str).str.strip(),
        "opening_qty": _amt(raw, "opening_qty"),
        "opening_val": _amt(raw, "opening_val").round(2),
        "inward_qty": _amt(raw, "inward_qty"),
        "inward_val": _amt(raw, "inward_val").round(2),
        "outward_qty": _amt(raw, "outward_qty"),
        "outward_val": _amt(raw, "outward_val").round(2),
        "closing_qty": _amt(raw, "closing_qty"),
        "closing_val": _amt(raw, "closing_val").round(2),
        "rate": _amt(raw, "rate"),
    })
    out["item_key"] = out["item"].map(norm_party)
    out = _keep_extras(raw, out)
    out = out[out["item"].str.len() > 0]
    return _tag(out, "inventory")


def load_conversion(cfg: Config) -> pd.DataFrame:
    raw = load_frames(cfg.source_patterns("conversion"), "conversion", cfg, ROOT)
    if raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "txn_date": _date(raw, "txn_date"),
        "voucher_no": _col(raw, "voucher_no", "").astype(str).str.strip(),
        "output_item": _col(raw, "output_item", "").map(display_party),
        "output_qty": _amt(raw, "output_qty"),
        "output_val": _amt(raw, "output_val").round(2),
        "input_item": _col(raw, "input_item", "").map(display_party),
        "input_qty": _amt(raw, "input_qty"),
        "input_val": _amt(raw, "input_val").round(2),
        "std_yield": _amt(raw, "std_yield"),
    })
    out["output_key"] = out["output_item"].map(norm_party)
    out["input_key"] = out["input_item"].map(norm_party)
    out = _keep_extras(raw, out)
    out = out[(out["output_qty"] != 0) | (out["input_qty"] != 0)]
    return _tag(out, "conversion")


# ------------------------------------------------------- GST return sheets

def _load_gst_summary(cfg: Config, source: str, kind: str) -> pd.DataFrame:
    raw = load_frames(cfg.source_patterns(source), "gst_summary", cfg, ROOT)
    if raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "period": _col(raw, "period", "").map(parse_period),
        "taxable": _amt(raw, "taxable").round(2),
        "cgst": _amt(raw, "cgst").round(2),
        "sgst": _amt(raw, "sgst").round(2),
        "igst": _amt(raw, "igst").round(2),
        "cess": _amt(raw, "cess").round(2),
        "itc_cgst": _amt(raw, "itc_cgst").round(2),
        "itc_sgst": _amt(raw, "itc_sgst").round(2),
        "itc_igst": _amt(raw, "itc_igst").round(2),
        "paid_cash": _amt(raw, "paid_cash").round(2),
        "paid_credit": _amt(raw, "paid_credit").round(2),
        "challan_no": _col(raw, "challan_no", "").astype(str).str.strip(),
        "paid_date": _date(raw, "paid_date"),
    })
    out["tax"] = (out["cgst"] + out["sgst"] + out["igst"] + out["cess"]).round(2)
    out["itc"] = (out["itc_cgst"] + out["itc_sgst"] + out["itc_igst"]).round(2)
    out = _keep_extras(raw, out)
    out = out[out["period"].notna()]
    return _tag(out, kind)


def load_gstr1(cfg: Config) -> pd.DataFrame:
    return _load_gst_summary(cfg, "gstr1", "gstr1")


def load_gstr3b(cfg: Config) -> pd.DataFrame:
    return _load_gst_summary(cfg, "gstr3b", "gstr3b")


# --------------------------------------------------------------- bundle

LOADERS = {
    "bank": load_bank,
    "bankbook": load_bankbook,
    "cashbook": load_cashbook,
    "purchase": load_purchase,
    "sales": load_sales,
    "gstr2b": load_gstr2b,
    "gstr1": load_gstr1,
    "gstr3b": load_gstr3b,
    "receipts": load_receipts,
    "payments": load_payments,
    "debtors": load_debtors,
    "creditors": load_creditors,
    "salary": load_salary,
    "inventory": load_inventory,
    "conversion": load_conversion,
}


def load_all(cfg: Config) -> dict[str, pd.DataFrame]:
    """Load every configured source. Missing sources yield empty frames so
    downstream modules can simply check ``.empty``."""
    data: dict[str, pd.DataFrame] = {}
    for name, fn in LOADERS.items():
        try:
            data[name] = fn(cfg)
        except Exception as exc:
            log.error("ingest failed for %s: %s", name, exc, exc_info=True)
            data[name] = pd.DataFrame()
    return data
