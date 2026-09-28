"""Bank reconciliation (BRS).

Statement vs bank book, then the classic four reconciling item classes:
cheques issued not presented, deposits not credited, bank-originated debits
(charges/EMI) not in books, bank-originated credits (interest/direct receipts)
not in books.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..matching import Pass, cascade_match
from ..normalize import r2

_PASSES = (
    # a matching cheque/UTR number is as good as it gets
    Pass("ref no + amount", keys=[("ref_no", "ref_no")],
         amount=("amount", "amount"), amount_tol=1.0, confidence=100),
    # same amount, same day
    Pass("amount + same date", amount=("amount", "amount"), amount_tol=1.0,
         date=("txn_date", "txn_date"), date_tol=0, confidence=98),
    # same amount within the clearing window, narration agrees
    Pass("amount + date + narration", amount=("amount", "amount"), amount_tol=1.0,
         date=("txn_date", "txn_date"), date_tol=7,
         fuzzy=("party_key", "party_key"), fuzzy_threshold=80, confidence=92),
    # last resort: amount alone inside the window
    Pass("amount + date window", amount=("amount", "amount"), amount_tol=1.0,
         date=("txn_date", "txn_date"), date_tol=10, confidence=75),
)

_BANK_ONLY_REASON = {
    "CHARGES": "Bank charges / fees not recorded in books",
    "INTEREST": "Bank interest not recorded in books",
    "TAX": "Direct tax debit not recorded in books",
    "LOAN": "EMI / loan debit not recorded in books",
    "CONTRA": "Contra entry not recorded in books",
}


def synth_book_side(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """If no bank book was supplied, build one from banked receipts/payments.

    Small offices often have no clean bank ledger export; receipts and
    payments vouchers with a non-cash mode are the same information.
    """
    parts = []
    for key, sign in (("receipts", 1), ("payments", -1)):
        df = data.get(key)
        if df is None or df.empty:
            continue
        sub = df[~df["is_cash"]].copy()
        if sub.empty:
            continue
        parts.append(pd.DataFrame({
            "row_id": sub["row_id"],
            "txn_date": sub["txn_date"],
            "description": sub["party"].astype(str) + " - " + sub["narration"].astype(str),
            "ref_no": sub["ref_no"],
            "amount": (sub["amount"] * sign).round(2),
            "party_key": sub["party_key"],
            "category": "TRADE",
            "dataset": key,
        }))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["debit"] = out["amount"].clip(lower=0)
    out["credit"] = (-out["amount"]).clip(lower=0)
    out["direction"] = np.where(out["amount"] >= 0, "INFLOW", "OUTFLOW")
    return out


def _closing(df: pd.DataFrame) -> float:
    """Closing balance: the statement's own last running balance if present,
    otherwise the net movement."""
    if df.empty:
        return 0.0
    if "balance" in df.columns and df["balance"].abs().sum() > 0:
        last = df.sort_values("txn_date").iloc[-1]["balance"]
        if pd.notna(last) and float(last) != 0:
            return r2(last)
    return r2(df["amount"].sum())


def run(data: dict[str, pd.DataFrame], cfg: Config) -> dict[str, pd.DataFrame]:
    bank = data.get("bank", pd.DataFrame())
    book = data.get("bankbook", pd.DataFrame())
    if book is None or book.empty:
        book = synth_book_side(data)

    opening = 0.0
    if not book.empty and "is_opening" in book.columns:
        opening = r2(book.loc[book["is_opening"], "amount"].sum())
        book = book[~book["is_opening"]].copy()

    if bank.empty or book.empty:
        return {"bank_summary": pd.DataFrame(), "bank_matched": pd.DataFrame(),
                "bank_only": pd.DataFrame(), "book_only": pd.DataFrame(),
                "brs": pd.DataFrame()}

    tol_days = int(cfg.th("date_tolerance_days", 7))
    passes = []
    for p in _PASSES:
        p = Pass(**{**p.__dict__})
        if p.date is not None:
            p.date_tol = min(p.date_tol, tol_days) if p.date_tol > 0 else 0
        p.amount_tol = float(cfg.th("amount_tolerance", 1.0))
        passes.append(p)

    res = cascade_match(bank, book, passes, "bank", "book")
    matched, bank_only, book_only = res["matched"], res["left_only"], res["right_only"]

    bank_only = bank_only.copy()
    fallback = pd.Series(np.where(bank_only["amount"] > 0,
                                  "Credit in bank not recorded in books",
                                  "Debit in bank not recorded in books"),
                         index=bank_only.index)
    bank_only["reason"] = bank_only["category"].map(_BANK_ONLY_REASON).fillna(fallback)

    book_only = book_only.copy()
    book_only["reason"] = np.where(
        book_only["amount"] < 0,
        "Cheque issued / payment entered but not presented",
        "Deposit entered but not credited by bank")
    asof = cfg.period_to
    book_only["days_pending"] = (asof - pd.to_datetime(book_only["txn_date"])).dt.days
    book_only["stale"] = book_only["days_pending"] > int(cfg.th("stale_cheque_days", 90))

    bank_close = _closing(bank)
    book_close = r2(opening + _closing(book))
    bo_in = r2(bank_only.loc[bank_only["amount"] > 0, "amount"].sum())
    bo_out = r2(-bank_only.loc[bank_only["amount"] < 0, "amount"].sum())
    bk_in = r2(book_only.loc[book_only["amount"] > 0, "amount"].sum())
    bk_out = r2(-book_only.loc[book_only["amount"] < 0, "amount"].sum())
    derived = r2(bank_close - bo_in + bo_out + bk_in - bk_out)

    brs = pd.DataFrame([
        ("A", "Balance as per bank statement", bank_close),
        ("B", "Less: credits in bank not in books (interest, direct receipts)", -bo_in),
        ("C", "Add: debits in bank not in books (charges, EMI, direct debits)", bo_out),
        ("D", "Add: deposits entered in books, not yet credited by bank", bk_in),
        ("E", "Less: cheques issued in books, not yet presented", -bk_out),
        ("F", "Balance as per books (derived: A+B+C+D+E)", derived),
        ("G", "Balance as per books (actual, incl. opening Rs.%s)" % f"{opening:,.2f}",
         book_close),
        ("H", "UNRECONCILED DIFFERENCE (F-G)", r2(derived - book_close)),
    ], columns=["Ref", "Particulars", "Amount"])

    rate = round(100.0 * len(matched) / max(len(bank), 1), 1)
    summary = pd.DataFrame([{
        "Bank statement lines": len(bank),
        "Book lines": len(book),
        "Matched": len(matched),
        "Match rate %": rate,
        "In bank, not in books": len(bank_only),
        "In books, not in bank": len(book_only),
        "Stale items (> %d days)" % int(cfg.th("stale_cheque_days", 90)):
            int(book_only["stale"].sum()),
        "Balance as per bank": bank_close,
        "Balance as per books": book_close,
        "Unreconciled difference": r2(derived - book_close),
    }])

    return {"bank_summary": summary, "bank_matched": matched, "bank_only": bank_only,
            "book_only": book_only, "brs": brs}
