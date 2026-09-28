"""The audit exception engine.

Every rule is a small function registered with :func:`rule`. It receives the
ingested data, the reconciliation results and the config, and returns a frame
of findings in one standard shape:

    code | rule | severity | category | dataset | row_id | date | party |
    amount | observation | reference | suggested_action

Adding a check means adding one function -- nothing else in the codebase
changes. Thresholds all come from config.yaml so the same rule set works for
a trader, a manufacturer and a service company.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import Config
from .matching import find_duplicates
from .normalize import inr, is_round_sum, r2, validate_pan

log = logging.getLogger("auditbot.exceptions")

COLUMNS = ["code", "rule", "severity", "category", "dataset", "row_id", "date",
           "party", "amount", "observation", "reference", "suggested_action"]


@dataclass
class Rule:
    code: str
    title: str
    severity: str
    category: str
    reference: str
    fn: Callable


REGISTRY: list[Rule] = []


def rule(code: str, title: str, severity: str = "MEDIUM", category: str = "General",
         reference: str = ""):
    def deco(fn: Callable) -> Callable:
        REGISTRY.append(Rule(code, title, severity, category, reference, fn))
        return fn
    return deco


def _mk(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[c for c in COLUMNS
                                       if c not in ("code", "rule", "severity",
                                                    "category", "reference")]) \
        if rows else pd.DataFrame()


def _df(df, cols: list[str]) -> bool:
    return df is not None and not df.empty and set(cols).issubset(df.columns)


def _rows(df: pd.DataFrame, *, dataset: str, date: str | None, party: str | None,
          amount: str | None, observation, action: str) -> list[dict]:
    """Project an arbitrary frame onto the exception shape."""
    out = []
    for _, r in df.iterrows():
        obs = observation(r) if callable(observation) else str(observation)
        out.append({
            "dataset": dataset,
            "row_id": r.get("row_id", ""),
            "date": r.get(date) if date else None,
            "party": r.get(party) if party else "",
            "amount": r2(r.get(amount, 0)) if amount else 0.0,
            "observation": obs,
            "suggested_action": action,
        })
    return out


# ====================================================================
#  CASH  (Income-tax Act limits)
# ====================================================================

@rule("CSH-01", "Cash payment above s.40A(3) limit", "HIGH", "Cash",
      "s.40A(3) Income-tax Act, 1961")
def cash_payment_limit(data, recon, cfg: Config) -> pd.DataFrame:
    limit = float(cfg.th("cash_payment_limit", 10000))
    rows = []
    pay = data.get("payments")
    if _df(pay, ["is_cash", "amount"]):
        hits = pay[pay["is_cash"] & (pay["amount"] > limit)]
        rows += _rows(hits, dataset="payments", date="txn_date", party="party",
                      amount="amount",
                      observation=lambda r: (f"Cash payment of Rs.{inr(r['amount'])} to "
                                             f"{r['party']} exceeds Rs.{inr(limit)}"),
                      action="Disallow under s.40A(3) unless Rule 6DD exception is "
                             "documented; obtain the payee declaration.")
    cash = data.get("cashbook")
    if _df(cash, ["amount"]):
        hits = cash[cash["amount"] < -limit]
        rows += _rows(hits, dataset="cashbook", date="txn_date", party="description",
                      amount="amount",
                      observation=lambda r: (f"Cash book payment of Rs.{inr(abs(r['amount']))} "
                                             f"- {r['description'][:60]}"),
                      action="Verify against Rule 6DD exceptions; consider disallowance.")
    return _mk(rows)


@rule("CSH-02", "Aggregate cash paid to one party on one day above limit", "HIGH",
      "Cash", "s.40A(3) - aggregate of payments in a day")
def cash_payment_aggregate(data, recon, cfg: Config) -> pd.DataFrame:
    limit = float(cfg.th("cash_payment_limit", 10000))
    pay = data.get("payments")
    if not _df(pay, ["is_cash", "amount", "party", "txn_date"]):
        return pd.DataFrame()
    cashpay = pay[pay["is_cash"]]
    if cashpay.empty:
        return pd.DataFrame()
    g = (cashpay.groupby(["party", "txn_date"])
                .agg(amount=("amount", "sum"), vouchers=("amount", "size"))
                .reset_index())
    hits = g[(g["amount"] > limit) & (g["vouchers"] > 1)]
    return _mk(_rows(hits, dataset="payments", date="txn_date", party="party",
                     amount="amount",
                     observation=lambda r: (f"{int(r['vouchers'])} cash vouchers totalling "
                                            f"Rs.{inr(r['amount'])} paid to {r['party']} on "
                                            f"one day - split below the limit"),
                     action="Aggregate per person per day; disallow under s.40A(3)."))


@rule("CSH-03", "Cash receipt at or above s.269ST limit", "HIGH", "Cash",
      "s.269ST / penalty s.271DA")
def cash_receipt_limit(data, recon, cfg: Config) -> pd.DataFrame:
    limit = float(cfg.th("cash_receipt_limit", 200000))
    rec = data.get("receipts")
    if not _df(rec, ["is_cash", "amount"]):
        return pd.DataFrame()
    hits = rec[rec["is_cash"] & (rec["amount"] >= limit)]
    return _mk(_rows(hits, dataset="receipts", date="txn_date", party="party",
                     amount="amount",
                     observation=lambda r: (f"Cash receipt of Rs.{inr(r['amount'])} from "
                                            f"{r['party']} is at/above Rs.{inr(limit)}"),
                     action="Penalty u/s 271DA equal to the amount received. Confirm "
                            "whether it relates to a single transaction or event."))


@rule("CSH-04", "Aggregate cash received from one party on one day at/above 269ST",
      "HIGH", "Cash", "s.269ST(a)")
def cash_receipt_aggregate(data, recon, cfg: Config) -> pd.DataFrame:
    limit = float(cfg.th("cash_receipt_limit", 200000))
    rec = data.get("receipts")
    if not _df(rec, ["is_cash", "amount", "party", "txn_date"]):
        return pd.DataFrame()
    cashrec = rec[rec["is_cash"]]
    if cashrec.empty:
        return pd.DataFrame()
    g = (cashrec.groupby(["party", "txn_date"])
                .agg(amount=("amount", "sum"), vouchers=("amount", "size")).reset_index())
    hits = g[(g["amount"] >= limit) & (g["vouchers"] > 1)]
    return _mk(_rows(hits, dataset="receipts", date="txn_date", party="party",
                     amount="amount",
                     observation=lambda r: (f"{int(r['vouchers'])} cash receipts totalling "
                                            f"Rs.{inr(r['amount'])} from {r['party']} in a day"),
                     action="Aggregate per person per day under s.269ST(a)."))


@rule("CSH-05", "Negative cash balance", "HIGH", "Cash",
      "Physical impossibility - indicates unrecorded receipts")
def negative_cash(data, recon, cfg: Config) -> pd.DataFrame:
    cash = data.get("cashbook")
    if not _df(cash, ["amount", "txn_date"]):
        return pd.DataFrame()
    d = cash.sort_values("txn_date").copy()
    d["running"] = d["amount"].cumsum().round(2)
    hits = d[d["running"] < -0.01]
    return _mk(_rows(hits, dataset="cashbook", date="txn_date", party="description",
                     amount="running",
                     observation=lambda r: (f"Cash balance goes negative to "
                                            f"Rs.{inr(r['running'])} on this date"),
                     action="Cash cannot be negative - trace unrecorded receipts or "
                            "mis-dated payments."))


# ====================================================================
#  DUPLICATES
# ====================================================================

@rule("DUP-01", "Duplicate purchase invoice", "HIGH", "Duplicates",
      "Same supplier and invoice number recorded more than once")
def dup_purchase(data, recon, cfg: Config) -> pd.DataFrame:
    p = data.get("purchase")
    if not _df(p, ["party_key", "inv_key"]):
        return pd.DataFrame()
    dups = find_duplicates(p[p["inv_key"].str.len() > 0], ["party_key", "inv_key"])
    return _mk(_rows(dups, dataset="purchase", date="txn_date", party="party",
                     amount="total",
                     observation=lambda r: (f"Invoice {r['invoice_no']} from {r['party']} "
                                            f"appears more than once (group "
                                            f"{int(r.get('dup_group', 0))})"),
                     action="Reverse the duplicate entry and the ITC claimed on it."))


@rule("DUP-02", "Duplicate sales invoice number", "HIGH", "Duplicates",
      "Same invoice number issued twice")
def dup_sales(data, recon, cfg: Config) -> pd.DataFrame:
    s = data.get("sales")
    if not _df(s, ["inv_key"]):
        return pd.DataFrame()
    dups = find_duplicates(s[s["inv_key"].str.len() > 0], ["inv_key"])
    return _mk(_rows(dups, dataset="sales", date="txn_date", party="party",
                     amount="total",
                     observation=lambda r: (f"Sales invoice number {r['invoice_no']} is used "
                                            f"more than once"),
                     action="Rule 46 requires a unique consecutive invoice number - "
                            "investigate and correct the series."))


@rule("DUP-03", "Possible duplicate payment", "HIGH", "Duplicates",
      "Same party, same amount, within a short window")
def dup_payment(data, recon, cfg: Config) -> pd.DataFrame:
    pay = data.get("payments")
    if not _df(pay, ["party_key", "amount", "txn_date"]):
        return pd.DataFrame()
    window = int(cfg.th("duplicate_date_window", 3))
    dups = find_duplicates(pay, ["party_key"], amount_col="amount",
                           date_col="txn_date", date_window=window)
    return _mk(_rows(dups, dataset="payments", date="txn_date", party="party",
                     amount="amount",
                     observation=lambda r: (f"Payment of Rs.{inr(r['amount'])} to {r['party']} "
                                            f"repeats within {window} days"),
                     action="Confirm it is not a double payment; recover if it is."))


@rule("DUP-04", "Possible duplicate receipt", "MEDIUM", "Duplicates",
      "Same party, same amount, within a short window")
def dup_receipt(data, recon, cfg: Config) -> pd.DataFrame:
    rec = data.get("receipts")
    if not _df(rec, ["party_key", "amount", "txn_date"]):
        return pd.DataFrame()
    window = int(cfg.th("duplicate_date_window", 3))
    dups = find_duplicates(rec, ["party_key"], amount_col="amount",
                           date_col="txn_date", date_window=window)
    return _mk(_rows(dups, dataset="receipts", date="txn_date", party="party",
                     amount="amount",
                     observation=lambda r: (f"Receipt of Rs.{inr(r['amount'])} from "
                                            f"{r['party']} repeats within {window} days"),
                     action="Confirm both receipts are genuine and separately banked."))


@rule("DUP-05", "Duplicate bank transaction", "MEDIUM", "Duplicates",
      "Identical narration, amount and date in the statement")
def dup_bank(data, recon, cfg: Config) -> pd.DataFrame:
    b = data.get("bank")
    if not _df(b, ["description", "amount", "txn_date"]):
        return pd.DataFrame()
    dups = find_duplicates(b, ["description", "txn_date"], amount_col="amount")
    return _mk(_rows(dups, dataset="bank", date="txn_date", party="description",
                     amount="amount",
                     observation=lambda r: (f"Identical bank line Rs.{inr(r['amount'])} - "
                                            f"{str(r['description'])[:60]}"),
                     action="Check whether the statement was imported twice."))


# ====================================================================
#  GST
# ====================================================================

@rule("GST-01", "ITC claimed but invoice not in GSTR-2B", "HIGH", "GST",
      "s.16(2)(aa) CGST Act - ITC only if reflected in GSTR-2B")
def itc_not_in_2b(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("itc_books_only")
    if d is None or d.empty:
        return pd.DataFrame()
    return _mk(_rows(d, dataset="purchase", date="txn_date", party="party", amount="tax",
                     observation=lambda r: (f"ITC of Rs.{inr(r['tax'])} on invoice "
                                            f"{r['invoice_no']} is not reflected in GSTR-2B"),
                     action="Follow up with the supplier to file GSTR-1, or reverse the ITC "
                            "with interest u/s 50."))


@rule("GST-02", "Invoice in GSTR-2B but not booked", "MEDIUM", "GST",
      "Unclaimed ITC / unrecorded purchase")
def purchase_not_booked(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("itc_portal_only")
    if d is None or d.empty:
        return pd.DataFrame()
    return _mk(_rows(d, dataset="gstr2b", date="txn_date", party="party", amount="tax",
                     observation=lambda r: (f"GSTR-2B shows invoice {r['invoice_no']} from "
                                            f"{r['party']} with ITC Rs.{inr(r['tax'])} not in books"),
                     action="Record the purchase and claim the ITC before the s.16(4) "
                            "time limit expires."))


@rule("GST-03", "Tax value differs between books and GSTR-2B", "MEDIUM", "GST",
      "Per-invoice tax mismatch")
def itc_value_mismatch(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("itc_value_diff")
    if d is None or d.empty:
        return pd.DataFrame()
    return _mk(_rows(d, dataset="purchase", date="books_txn_date", party="books_party",
                     amount="tax_diff",
                     observation=lambda r: (f"Invoice {r.get('books_invoice_no')}: books tax "
                                            f"differs from GSTR-2B by Rs.{inr(r['tax_diff'])}"),
                     action="Agree the invoice with the supplier copy and correct the "
                            "lower of the two."))


@rule("GST-04", "GSTR-3B short paid", "HIGH", "GST",
      "s.50 CGST Act - interest on delayed payment")
def gst_short_paid(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("gst_payment")
    if d is None or d.empty or "status" not in d.columns:
        return pd.DataFrame()
    hits = d[d["status"] == "SHORT PAID"].copy()
    hits["row_id"] = hits["period"]
    return _mk(_rows(hits, dataset="gstr3b", date="paid_date", party="period",
                     amount="shortfall",
                     observation=lambda r: (f"Period {r['period']}: liability Rs."
                                            f"{inr(r['liability'])} vs paid Rs."
                                            f"{inr(r['paid_total'])} - short by "
                                            f"Rs.{inr(r['shortfall'])}"),
                     action="Pay the differential with interest u/s 50 through DRC-03."))


@rule("GST-05", "Books outward supply differs from GSTR-3B", "HIGH", "GST",
      "Turnover under-reported / over-reported in the return")
def books_vs_3b(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("gst_books_vs_3b")
    if d is None or d.empty:
        return pd.DataFrame()
    tol = float(cfg.th("tax_tolerance", 5.0))
    hits = d[d[["diff_taxable", "diff_tax"]].abs().max(axis=1) > tol].copy()
    hits["row_id"] = hits["period"]
    return _mk(_rows(hits, dataset="gstr3b", date=None, party="period",
                     amount="diff_tax",
                     observation=lambda r: (f"Period {r['period']}: books taxable Rs."
                                            f"{inr(r['books_taxable'])} vs GSTR-3B Rs."
                                            f"{inr(r['gstr3b_taxable'])} "
                                            f"(tax difference Rs.{inr(r['diff_tax'])})"),
                     action="Reconcile and report the difference in GSTR-9/9C; amend the "
                            "return if still open."))


@rule("GST-06", "Books outward supply differs from GSTR-1", "MEDIUM", "GST",
      "GSTR-1 vs books turnover")
def books_vs_gstr1(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("sales_monthly")
    if d is None or d.empty or "status" not in d.columns:
        return pd.DataFrame()
    hits = d[d["status"] == "MISMATCH"].copy()
    hits["row_id"] = hits["period"]
    return _mk(_rows(hits, dataset="gstr1", date=None, party="period", amount="diff_tax",
                     observation=lambda r: (f"Period {r['period']}: books tax Rs."
                                            f"{inr(r['books_tax'])} vs GSTR-1 Rs."
                                            f"{inr(r['gstr1_tax'])}"),
                     action="Amend GSTR-1 for the missing/extra invoices."))


@rule("GST-07", "GSTIN invalid or tax head inconsistent with place of supply",
      "MEDIUM", "GST", "s.31 / Rule 46 - invoice particulars")
def gstin_issues(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("gstin_issues")
    if d is None or d.empty:
        return pd.DataFrame()
    return _mk(_rows(d, dataset="invoices", date="date", party="party", amount="total",
                     observation=lambda r: f"{r['side']} invoice {r['invoice_no']}: {r['issue']}",
                     action="Correct the GSTIN / tax head; an invalid GSTIN blocks the "
                            "counter-party's ITC."))


@rule("GST-08", "Cash paid on GST although ITC was available", "LOW", "GST",
      "s.49 - order of utilisation")
def gst_cash_with_itc(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("gst_payment")
    if d is None or d.empty:
        return pd.DataFrame()
    hits = d[(d["paid_cash"] > 0) & (d["itc_available"] > d["paid_credit"] + 1)].copy()
    if hits.empty:
        return pd.DataFrame()
    hits["row_id"] = hits["period"]
    hits["unused"] = (hits["itc_available"] - hits["paid_credit"]).round(2)
    return _mk(_rows(hits, dataset="gstr3b", date="paid_date", party="period",
                     amount="paid_cash",
                     observation=lambda r: (f"Period {r['period']}: Rs.{inr(r['paid_cash'])} "
                                            f"paid in cash while Rs.{inr(r['unused'])} of ITC "
                                            f"stayed unutilised"),
                     action="Check the utilisation order - avoidable cash outflow and "
                            "blocked working capital."))


@rule("GST-09", "Challan reference missing for GST paid in cash", "LOW", "GST",
      "Audit trail for tax payments")
def gst_challan_missing(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("gst_payment")
    if d is None or d.empty or "challan_missing" not in d.columns:
        return pd.DataFrame()
    hits = d[d["challan_missing"]].copy()
    hits["row_id"] = hits["period"]
    return _mk(_rows(hits, dataset="gstr3b", date="paid_date", party="period",
                     amount="paid_cash",
                     observation=lambda r: (f"Period {r['period']}: Rs.{inr(r['paid_cash'])} "
                                            f"paid in cash without a challan/CIN reference"),
                     action="Obtain the challan (CIN/BRN) and file it in the working papers."))


# ====================================================================
#  BANK
# ====================================================================

@rule("BNK-01", "Bank entry not recorded in books", "HIGH", "Bank",
      "Completeness of the cash book")
def bank_unmatched(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("bank_only")
    if d is None or d.empty:
        return pd.DataFrame()
    return _mk(_rows(d, dataset="bank", date="txn_date", party="description",
                     amount="amount",
                     observation=lambda r: (f"{r['reason']}: Rs.{inr(abs(r['amount']))} - "
                                            f"{str(r['description'])[:60]}"),
                     action="Pass the missing entry in the books before finalising."))


@rule("BNK-02", "Cheque outstanding beyond validity", "HIGH", "Bank",
      "Cheque validity is 3 months from the date of issue")
def stale_cheques(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("book_only")
    if d is None or d.empty or "stale" not in d.columns:
        return pd.DataFrame()
    hits = d[d["stale"]]
    return _mk(_rows(hits, dataset="bankbook", date="txn_date", party="description",
                     amount="amount",
                     observation=lambda r: (f"Entered in books {int(r['days_pending'])} days ago "
                                            f"and still not through the bank - "
                                            f"Rs.{inr(abs(r['amount']))}"),
                     action="Reverse stale cheques and write back the liability."))


@rule("BNK-03", "Bank reconciliation does not tie", "HIGH", "Bank",
      "BRS must reconcile to nil")
def brs_difference(data, recon, cfg: Config) -> pd.DataFrame:
    brs = recon.get("brs")
    if brs is None or brs.empty:
        return pd.DataFrame()
    row = brs[brs["Ref"] == "H"]
    if row.empty or abs(float(row.iloc[0]["Amount"])) <= 1.0:
        return pd.DataFrame()
    diff = float(row.iloc[0]["Amount"])
    return _mk([{"dataset": "bank", "row_id": "BRS", "date": cfg.period_to,
                 "party": "Bank reconciliation", "amount": r2(diff),
                 "observation": f"Unreconciled difference of Rs.{inr(diff)} between the "
                                f"derived and actual book balance",
                 "suggested_action": "Investigate before signing off - the BRS must "
                                     "reconcile to nil."}])


@rule("BNK-04", "Bank charges or interest not accounted", "MEDIUM", "Bank",
      "Expenses debited by the bank must be booked")
def bank_charges(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("bank_only")
    if d is None or d.empty:
        return pd.DataFrame()
    hits = d[d["category"].isin(["CHARGES", "INTEREST"]) & (d["amount"] < 0)]
    return _mk(_rows(hits, dataset="bank", date="txn_date", party="description",
                     amount="amount",
                     observation=lambda r: (f"{r['category'].title()} of "
                                            f"Rs.{inr(abs(r['amount']))} debited by the bank "
                                            f"is not in the books"),
                     action="Book the charge and claim the GST credit on it where eligible."))


# ====================================================================
#  DEBTORS / CREDITORS
# ====================================================================

@rule("DBT-01", "Debtor outstanding beyond 180 days", "HIGH", "Debtors",
      "Recoverability / provision for doubtful debts")
def old_debtors(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("debtors_detail")
    if d is None or d.empty:
        return pd.DataFrame()
    hits = d[d["age_days"] > 180]
    return _mk(_rows(hits, dataset="debtors", date="bill_date", party="party",
                     amount="balance",
                     observation=lambda r: (f"Bill {r['bill_no']} of Rs.{inr(r['balance'])} "
                                            f"is {int(r['age_days'])} days old"),
                     action="Obtain balance confirmation and assess a provision for "
                            "doubtful debt."))


@rule("DBT-02", "Debtor beyond the agreed credit period", "MEDIUM", "Debtors",
      "Credit control")
def overdue_debtors(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("debtors_detail")
    if d is None or d.empty:
        return pd.DataFrame()
    hits = d[(d["overdue_days"] > 0) & (d["age_days"] <= 180)]
    return _mk(_rows(hits, dataset="debtors", date="due_date", party="party",
                     amount="balance",
                     observation=lambda r: (f"Bill {r['bill_no']} overdue by "
                                            f"{int(r['overdue_days'])} days "
                                            f"(Rs.{inr(r['balance'])})"),
                     action="Follow up for collection; review the credit limit."))


@rule("DBT-03", "Credit balance in a debtor account", "MEDIUM", "Debtors",
      "Advance from customer shown as a debtor - Schedule III regrouping")
def credit_debtors(data, recon, cfg: Config) -> pd.DataFrame:
    d = data.get("debtors")
    if not _df(d, ["balance"]):
        return pd.DataFrame()
    hits = d[d["balance"] < -0.01]
    return _mk(_rows(hits, dataset="debtors", date="bill_date", party="party",
                     amount="balance",
                     observation=lambda r: (f"{r['party']} shows a credit balance of "
                                            f"Rs.{inr(abs(r['balance']))}"),
                     action="Regroup to advances from customers in the balance sheet."))


@rule("CRD-01", "Supplier unpaid beyond 180 days - ITC reversal", "HIGH", "Creditors",
      "Second proviso to s.16(2) / Rule 37 CGST Rules")
def creditor_itc_reversal(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("creditors_detail")
    if d is None or d.empty:
        return pd.DataFrame()
    days = int(cfg.th("creditor_itc_reversal_days", 180))
    hits = d[d["age_days"] > days]
    return _mk(_rows(hits, dataset="creditors", date="bill_date", party="party",
                     amount="balance",
                     observation=lambda r: (f"Bill {r['bill_no']} of Rs.{inr(r['balance'])} "
                                            f"unpaid for {int(r['age_days'])} days"),
                     action=f"Reverse the ITC with interest - supplier unpaid beyond "
                            f"{days} days (Rule 37)."))


@rule("CRD-02", "Debit balance in a creditor account", "MEDIUM", "Creditors",
      "Advance to supplier shown as a creditor - Schedule III regrouping")
def debit_creditors(data, recon, cfg: Config) -> pd.DataFrame:
    d = data.get("creditors")
    if not _df(d, ["balance"]):
        return pd.DataFrame()
    hits = d[d["balance"] < -0.01]
    return _mk(_rows(hits, dataset="creditors", date="bill_date", party="party",
                     amount="balance",
                     observation=lambda r: (f"{r['party']} shows a debit balance of "
                                            f"Rs.{inr(abs(r['balance']))}"),
                     action="Regroup to advances to suppliers / other current assets."))


@rule("CRD-03", "Party balance concentrated in a few accounts", "LOW", "Debtors",
      "Concentration risk disclosure")
def concentration(data, recon, cfg: Config) -> pd.DataFrame:
    rows = []
    for kind in ("debtors", "creditors"):
        p = recon.get(f"{kind}_party")
        if p is None or p.empty or len(p) < 5:
            continue
        top5 = float(p["% of total"].head(5).sum())
        if top5 < 60:
            continue
        rows.append({
            "dataset": kind, "row_id": f"{kind.upper()}-CONC",
            "date": cfg.period_to, "party": f"Top 5 {kind}",
            "amount": r2(p["outstanding"].head(5).sum()),
            "observation": f"Top 5 {kind} account for {top5:.1f}% of the total balance",
            "suggested_action": "Consider a concentration-risk disclosure and obtain "
                                "confirmations from these parties.",
        })
    return _mk(rows)


# ====================================================================
#  BILL-WISE
# ====================================================================

@rule("BIL-01", "Receipt could not be applied to any bill", "MEDIUM", "Bill-wise",
      "On-account receipts distort the ageing")
def unapplied_receipts(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("receipts_unapplied")
    if d is None or d.empty:
        return pd.DataFrame()
    return _mk(_rows(d, dataset="receipts", date="txn_date", party="party",
                     amount="unapplied_amount",
                     observation=lambda r: (f"Rs.{inr(r['unapplied_amount'])} received from "
                                            f"{r['party']} remains on account"),
                     action="Obtain the bill reference from the party and knock it off."))


@rule("BIL-02", "Payment could not be applied to any bill", "MEDIUM", "Bill-wise",
      "On-account payments distort the ageing")
def unapplied_payments(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("payments_unapplied")
    if d is None or d.empty:
        return pd.DataFrame()
    return _mk(_rows(d, dataset="payments", date="txn_date", party="party",
                     amount="unapplied_amount",
                     observation=lambda r: (f"Rs.{inr(r['unapplied_amount'])} paid to "
                                            f"{r['party']} is not against any open bill"),
                     action="Identify the bill or treat it as an advance."))


# ====================================================================
#  SALARY
# ====================================================================

@rule("SAL-01", "Net pay does not equal gross less deductions", "HIGH", "Salary",
      "Payroll arithmetic")
def salary_arithmetic(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("salary_detail")
    if d is None or d.empty:
        return pd.DataFrame()
    hits = d[d["net_variance"].abs() > 1]
    return _mk(_rows(hits, dataset="salary", date=None, party="emp_name",
                     amount="net_variance",
                     observation=lambda r: (f"{r['emp_name']} ({r['month']}): net paid "
                                            f"Rs.{inr(r['net'])} vs computed "
                                            f"Rs.{inr(r['computed_net'])}"),
                     action="Recheck the payroll working for this employee-month."))


@rule("SAL-02", "PF deduction differs from the statutory computation", "MEDIUM",
      "Salary", "EPF Act - 12% of wages up to the ceiling")
def pf_variance(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("salary_detail")
    if d is None or d.empty:
        return pd.DataFrame()
    hits = d[d["pf_variance"].abs() > 1]
    return _mk(_rows(hits, dataset="salary", date=None, party="emp_name",
                     amount="pf_variance",
                     observation=lambda r: (f"{r['emp_name']} ({r['month']}): PF deducted "
                                            f"Rs.{inr(r['pf_employee'])} vs expected "
                                            f"Rs.{inr(r['pf_expected'])}"),
                     action="Reconcile with the ECR filed; short deduction attracts "
                            "damages u/s 14B."))


@rule("SAL-03", "ESI not deducted for an employee below the wage ceiling", "MEDIUM",
      "Salary", "ESI Act - coverage up to the wage ceiling")
def esi_missing(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("salary_detail")
    if d is None or d.empty:
        return pd.DataFrame()
    hits = d[d["esi_applicable"] & (d["esi_employee"] <= 0) & (d["gross"] > 0)]
    return _mk(_rows(hits, dataset="salary", date=None, party="emp_name",
                     amount="gross",
                     observation=lambda r: (f"{r['emp_name']} ({r['month']}): gross "
                                            f"Rs.{inr(r['gross'])} is within the ESI ceiling "
                                            f"but no ESI was deducted"),
                     action="Verify coverage; deposit the contribution with interest if due."))


@rule("SAL-04", "Salary paid in cash above the limit", "HIGH", "Salary",
      "s.40A(3) applies to salary payments as well")
def salary_cash(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("salary_detail")
    if d is None or d.empty:
        return pd.DataFrame()
    limit = float(cfg.th("cash_payment_limit", 10000))
    hits = d[d["is_cash"] & (d["net"] > limit)]
    return _mk(_rows(hits, dataset="salary", date=None, party="emp_name", amount="net",
                     observation=lambda r: (f"{r['emp_name']} ({r['month']}) paid "
                                            f"Rs.{inr(r['net'])} in cash"),
                     action="Disallow u/s 40A(3) and move the employee to bank transfer."))


@rule("SAL-05", "Two employees share a bank account, PAN or UAN", "HIGH", "Salary",
      "Ghost employee / payroll fraud indicator")
def salary_shared_identity(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("salary_employee")
    if d is None or d.empty:
        return pd.DataFrame()
    rows = []
    for field, label in (("bank_acct", "bank account"), ("pan", "PAN"), ("uan", "UAN")):
        if field not in d.columns:
            continue
        sub = d[d[field].astype(str).str.len() > 3]
        dup = sub[sub.duplicated(subset=[field], keep=False)]
        for _, r in dup.iterrows():
            rows.append({
                "dataset": "salary", "row_id": r["emp_key"], "date": None,
                "party": r["emp_name"], "amount": r2(r["gross"]),
                "observation": f"{r['emp_name']} shares {label} {r[field]} with another employee",
                "suggested_action": "Verify the employees exist and are distinct - classic "
                                    "ghost-employee indicator.",
            })
    return _mk(rows)


@rule("SAL-06", "Employee without a valid PAN", "MEDIUM", "Salary",
      "s.206AA - higher TDS rate where PAN is not furnished")
def salary_no_pan(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("salary_employee")
    if d is None or d.empty or "pan_valid" not in d.columns:
        return pd.DataFrame()
    limit = 250000
    hits = d[(~d["pan_valid"]) & (d["gross"] > limit)]
    return _mk(_rows(hits, dataset="salary", date=None, party="emp_name", amount="gross",
                     observation=lambda r: (f"{r['emp_name']} earned Rs.{inr(r['gross'])} "
                                            f"without a valid PAN on record"),
                     action="Obtain the PAN; otherwise deduct TDS at the higher rate u/s 206AA."))


@rule("SAL-07", "Payroll cost does not tie to the bank", "MEDIUM", "Salary",
      "Payroll to bank tie-up")
def salary_bank_diff(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("salary_bank_tieup")
    if d is None or d.empty:
        return pd.DataFrame()
    hits = d[d["status"] == "DIFFERENCE"].copy()
    hits["row_id"] = hits["month"]
    return _mk(_rows(hits, dataset="salary", date=None, party="month",
                     amount="difference",
                     observation=lambda r: (f"{r['month']}: payroll net Rs."
                                            f"{inr(r['payroll_net'])} vs salary debited in bank "
                                            f"Rs.{inr(r['bank_salary_debit'])}"),
                     action="Reconcile the difference - unpaid salary, cash top-ups or "
                            "misclassified bank entries."))


@rule("SAL-08", "Unusual month-on-month swing in an employee's pay", "LOW", "Salary",
      "Analytical review")
def salary_swing(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("salary_employee")
    if d is None or d.empty:
        return pd.DataFrame()
    pct = float(cfg.th("salary_variance_pct", 20.0))
    hits = d[(d["gross_swing %"] > pct) & (d["months_paid"] > 1)]
    return _mk(_rows(hits, dataset="salary", date=None, party="emp_name", amount="gross",
                     observation=lambda r: (f"{r['emp_name']}: monthly gross ranges from "
                                            f"Rs.{inr(r['min_gross'])} to Rs.{inr(r['max_gross'])} "
                                            f"({r['gross_swing %']:.0f}% swing)"),
                     action="Check for arrears, bonus, LOP or an incorrect master rate."))


# ====================================================================
#  INVENTORY
# ====================================================================

@rule("INV-01", "Negative closing stock", "HIGH", "Inventory",
      "Stock cannot be negative - purchases not booked or excess issues")
def negative_stock(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("inventory_detail")
    if d is None or d.empty:
        return pd.DataFrame()
    hits = d[d["negative_stock"] | d["negative_value"]]
    return _mk(_rows(hits, dataset="inventory", date=None, party="item",
                     amount="closing_val",
                     observation=lambda r: (f"{r['item']}: closing quantity "
                                            f"{r['closing_qty']} {r['uom']}, value "
                                            f"Rs.{inr(r['closing_val'])}"),
                     action="Trace unrecorded purchases or excess issues before closing "
                            "the books."))


@rule("INV-02", "Closing stock does not agree to the movement", "HIGH", "Inventory",
      "Opening + inward - outward must equal closing")
def stock_movement(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("inventory_detail")
    if d is None or d.empty:
        return pd.DataFrame()
    hits = d[d["qty_variance"].abs() > 0.001]
    return _mk(_rows(hits, dataset="inventory", date=None, party="item",
                     amount="val_variance",
                     observation=lambda r: (f"{r['item']}: closing {r['closing_qty']} vs "
                                            f"derived {r['derived_closing_qty']} "
                                            f"(variance {r['qty_variance']} {r['uom']})"),
                     action="Reconcile the stock ledger with the physical count sheet."))


@rule("INV-03", "Conversion yield outside tolerance", "MEDIUM", "Inventory",
      "Abnormal loss / unrecorded production")
def yield_variance(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("conversion_detail")
    if d is None or d.empty:
        return pd.DataFrame()
    hits = d[d["yield_flag"] != "NORMAL"]
    return _mk(_rows(hits, dataset="conversion", date="txn_date", party="output_item",
                     amount="output_val",
                     observation=lambda r: (f"Voucher {r['voucher_no']}: actual yield "
                                            f"{r['actual_yield %']:.1f}% vs standard "
                                            f"{r['std_yield %']:.1f}% ({r['yield_flag'].lower()})"),
                     action="Investigate abnormal loss, unrecorded by-product or "
                            "unrecorded production."))


@rule("INV-04", "Slow moving or non-moving stock", "MEDIUM", "Inventory",
      "AS-2 / Ind AS 2 - value at the lower of cost and net realisable value")
def slow_moving(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("inventory_detail")
    if d is None or d.empty:
        return pd.DataFrame()
    hits = d[(d["turnover_ratio"].fillna(0) < 0.2) & (d["closing_val"] > 0)]
    return _mk(_rows(hits, dataset="inventory", date=None, party="item",
                     amount="closing_val",
                     observation=lambda r: (f"{r['item']}: turnover ratio "
                                            f"{r['turnover_ratio']:.2f}, closing value "
                                            f"Rs.{inr(r['closing_val'])}"),
                     action="Assess net realisable value and provide for obsolescence."))


@rule("INV-05", "Closing valuation rate far from the purchase rate", "LOW", "Inventory",
      "Valuation consistency")
def rate_variance(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("inventory_detail")
    if d is None or d.empty:
        return pd.DataFrame()
    hits = d[d["rate_variance %"].abs() > 15]
    return _mk(_rows(hits, dataset="inventory", date=None, party="item",
                     amount="closing_val",
                     observation=lambda r: (f"{r['item']}: closing rate Rs.{inr(r['closing_rate'])} "
                                            f"vs purchase rate Rs.{inr(r['inward_rate'])} "
                                            f"({r['rate_variance %']:.1f}%)"),
                     action="Confirm the valuation method has been applied consistently."))


# ====================================================================
#  ANALYTICAL / JOURNAL-LEVEL
# ====================================================================

@rule("ANL-01", "Round-sum entry", "LOW", "Analytical",
      "Window dressing / estimated provision indicator")
def round_sums(data, recon, cfg: Config) -> pd.DataFrame:
    mult = float(cfg.th("round_sum_multiple", 100000))
    rows = []
    for key, amt, party in (("payments", "amount", "party"),
                            ("receipts", "amount", "party"),
                            ("sales", "total", "party"),
                            ("purchase", "total", "party")):
        df = data.get(key)
        if not _df(df, [amt]):
            continue
        hits = df[df[amt].map(lambda v: is_round_sum(v, mult))]
        rows += _rows(hits, dataset=key, date="txn_date", party=party, amount=amt,
                      observation=lambda r, a=amt: (f"Exactly round amount of "
                                                    f"Rs.{inr(r[a])} recorded"),
                      action="Round figures are rarely commercial - vouch to the "
                             "supporting document.")
    return _mk(rows)


@rule("ANL-02", "Entry posted close to the year end", "MEDIUM", "Analytical",
      "Cut-off testing")
def yearend_entries(data, recon, cfg: Config) -> pd.DataFrame:
    window = int(cfg.th("yearend_window_days", 7))
    material = float(cfg.th("material_amount", 100000))
    cutoff = cfg.period_to - pd.Timedelta(days=window)
    rows = []
    for key, amt in (("sales", "total"), ("purchase", "total"),
                     ("payments", "amount"), ("receipts", "amount")):
        df = data.get(key)
        if not _df(df, [amt, "txn_date"]):
            continue
        hits = df[(df["txn_date"] >= cutoff) & (df["txn_date"] <= cfg.period_to)
                  & (df[amt].abs() >= material)]
        rows += _rows(hits, dataset=key, date="txn_date", party="party", amount=amt,
                      observation=lambda r, a=amt, k=key: (
                          f"Material {k} entry of Rs.{inr(r[a])} within {window} days "
                          f"of the year end"),
                      action="Cut-off testing - confirm the goods/services actually "
                             "moved before the year end.")
    return _mk(rows)


@rule("ANL-03", "Money moved on a Sunday or a declared holiday", "LOW", "Analytical",
      "Unusual posting date")
def holiday_entries(data, recon, cfg: Config) -> pd.DataFrame:
    """Sunday *invoices* are ordinary in Indian trade and flagging them buries
    the register in noise, so the Sunday test is limited to money actually
    moving. A date on the client's declared holiday list is flagged wherever
    it appears."""
    holidays = cfg.holidays
    material = float(cfg.th("material_amount", 100000))
    rows = []
    money = {"payments", "receipts", "bank", "cashbook"}
    for key, amt in (("sales", "total"), ("purchase", "total"),
                     ("payments", "amount"), ("receipts", "amount"),
                     ("cashbook", "amount")):
        df = data.get(key)
        if not _df(df, [amt, "txn_date"]):
            continue
        mask = df["txn_date"].isin(holidays)
        if key in money:
            mask = mask | (df["txn_date"].dt.dayofweek == 6)
        hits = df[mask & (df[amt].abs() >= material)]
        rows += _rows(hits, dataset=key,
                      date="txn_date",
                      party="party" if "party" in hits.columns else "description",
                      amount=amt,
                      observation=lambda r, a=amt: (
                          f"Rs.{inr(r[a])} moved on "
                          f"{pd.Timestamp(r['txn_date']).strftime('%A, %d-%b-%Y')}"),
                      action="Confirm the transaction genuinely took place on a "
                             "non-working day.")
    return _mk(rows)


@rule("ANL-04", "Missing narration or bill reference", "LOW", "Analytical",
      "Audit trail - Rule 3(1) of the Companies (Accounts) Rules")
def missing_narration(data, recon, cfg: Config) -> pd.DataFrame:
    material = float(cfg.th("material_amount", 100000))
    rows = []
    for key in ("payments", "receipts"):
        df = data.get(key)
        if not _df(df, ["amount"]):
            continue
        blank = (df["narration"].astype(str).str.strip().str.len() < 3) \
            if "narration" in df.columns else pd.Series(True, index=df.index)
        noref = (df["bill_no"].astype(str).str.strip().str.len() < 1) \
            if "bill_no" in df.columns else pd.Series(True, index=df.index)
        hits = df[blank & noref & (df["amount"] >= material)]
        rows += _rows(hits, dataset=key, date="txn_date", party="party", amount="amount",
                      observation=lambda r: (f"Rs.{inr(r['amount'])} with no narration and "
                                             f"no bill reference"),
                      action="Record a narration and the bill reference - required for "
                             "the audit trail.")
    return _mk(rows)


@rule("ANL-05", "Gap or repeat in the sales invoice series", "MEDIUM", "Analytical",
      "Rule 46 CGST Rules - consecutive serial numbering")
def invoice_gaps(data, recon, cfg: Config) -> pd.DataFrame:
    d = recon.get("invoice_sequence")
    if d is None or d.empty:
        return pd.DataFrame()
    hits = d[(d["missing_count"] > 0) | (d["repeated_numbers"].astype(str).str.len() > 0)]
    rows = []
    for _, r in hits.iterrows():
        bits = []
        if r["missing_count"]:
            bits.append(f"{int(r['missing_count'])} numbers missing ({r['missing_numbers']})")
        if str(r["repeated_numbers"]):
            bits.append(f"repeated: {r['repeated_numbers']}")
        rows.append({
            "dataset": "sales", "row_id": f"SEQ-{r['series']}", "date": None,
            "party": r["series"], "amount": 0.0,
            "observation": f"Series {r['series']} runs {r['from']}-{r['to']}: " + "; ".join(bits),
            "suggested_action": "Produce the cancelled invoices for the missing numbers.",
        })
    return _mk(rows)


@rule("ANL-06", "Party crosses a TDS threshold", "HIGH", "Analytical",
      "Chapter XVII-B - TDS")
def tds_thresholds(data, recon, cfg: Config) -> pd.DataFrame:
    tds_cfg = cfg.get("thresholds.tds", {}) or {}
    pay = data.get("payments")
    if not _df(pay, ["party", "amount"]):
        return pd.DataFrame()
    annual = (pay.groupby("party")["amount"].agg(["sum", "size", "max"])
                 .rename(columns={"sum": "total", "size": "count", "max": "largest"})
                 .reset_index())
    rows = []
    limits = {k: float(v) for k, v in tds_cfg.items()}
    single = limits.get("194C_single", 30000)
    annual_limit = limits.get("194C_annual", 100000)
    for _, r in annual.iterrows():
        triggers = []
        if r["largest"] > single:
            triggers.append(f"single payment Rs.{inr(r['largest'])} > Rs.{inr(single)} (194C)")
        if r["total"] > annual_limit:
            triggers.append(f"annual Rs.{inr(r['total'])} > Rs.{inr(annual_limit)} (194C)")
        for sec in ("194J", "194I_rent", "194H", "194Q_purchase"):
            if sec in limits and r["total"] > limits[sec]:
                triggers.append(f"annual Rs.{inr(r['total'])} > Rs.{inr(limits[sec])} "
                                f"({sec.split('_')[0]})")
        if triggers:
            rows.append({
                "dataset": "payments", "row_id": f"TDS-{r['party'][:20]}", "date": None,
                "party": r["party"], "amount": r2(r["total"]),
                "observation": f"{r['party']}: " + "; ".join(triggers[:3]),
                "suggested_action": "Confirm the nature of the payment and that TDS was "
                                    "deducted and deposited; 30% disallowance u/s 40(a)(ia) "
                                    "otherwise.",
            })
    return _mk(rows)


@rule("ANL-07", "Party in the ledger has no GSTIN on any invoice", "LOW", "Analytical",
      "Vendor master completeness")
def missing_gstin(data, recon, cfg: Config) -> pd.DataFrame:
    material = float(cfg.th("material_amount", 100000))
    rows = []
    for key in ("purchase", "sales"):
        df = data.get(key)
        if not _df(df, ["gstin", "total", "party"]):
            continue
        g = df.assign(_has=df["gstin"].astype(str).str.len() == 15) \
              .groupby("party").agg(total=("total", "sum"), has=("_has", "max")).reset_index()
        hits = g[(~g["has"]) & (g["total"] >= material)]
        for _, r in hits.iterrows():
            rows.append({
                "dataset": key, "row_id": f"GSTIN-{r['party'][:20]}", "date": None,
                "party": r["party"], "amount": r2(r["total"]),
                "observation": f"{r['party']}: Rs.{inr(r['total'])} of {key} with no GSTIN "
                               f"on any invoice",
                "suggested_action": "Confirm the party is genuinely unregistered; check "
                                    "reverse-charge applicability.",
            })
    return _mk(rows)


# ====================================================================
#  build
# ====================================================================

def build_register(data: dict[str, pd.DataFrame], recon: dict[str, pd.DataFrame],
                   cfg: Config) -> pd.DataFrame:
    """Run every registered rule and return one ranked exception register."""
    frames: list[pd.DataFrame] = []
    for r in REGISTRY:
        try:
            out = r.fn(data, recon, cfg)
        except Exception as exc:
            log.error("rule %s failed: %s", r.code, exc, exc_info=True)
            continue
        if out is None or out.empty:
            continue
        out = out.copy()
        out["code"] = r.code
        out["rule"] = r.title
        out["severity"] = r.severity
        out["category"] = r.category
        out["reference"] = r.reference
        # keep the date column typed so concat does not have to guess
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        frames.append(out[[c for c in COLUMNS if c in out.columns]])

    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    reg = pd.concat(frames, ignore_index=True, sort=False)
    order = {s: i for i, s in enumerate(cfg.get("report.severity_order",
                                                ["HIGH", "MEDIUM", "LOW"]))}
    reg["_sev"] = reg["severity"].map(order).fillna(9)
    reg["_abs"] = reg["amount"].abs()
    reg = (reg.sort_values(["_sev", "_abs"], ascending=[True, False])
              .drop(columns=["_sev", "_abs"]).reset_index(drop=True))
    reg.insert(0, "sl", range(1, len(reg) + 1))
    return reg[["sl"] + COLUMNS]


def register_summary(reg: pd.DataFrame) -> pd.DataFrame:
    """Category x severity roll-up used on the dashboard."""
    if reg is None or reg.empty:
        return pd.DataFrame()
    g = (reg.groupby(["category", "severity"])
            .agg(count=("code", "size"), value=("amount", lambda s: r2(s.abs().sum())))
            .reset_index())
    pivot = g.pivot_table(index="category", columns="severity", values="count",
                          fill_value=0, aggfunc="sum")
    for col in ("HIGH", "MEDIUM", "LOW"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["HIGH", "MEDIUM", "LOW"]].reset_index()
    pivot["Total"] = pivot[["HIGH", "MEDIUM", "LOW"]].sum(axis=1)
    value = g.groupby("category")["value"].sum().round(2).rename("Value involved")
    return pivot.merge(value, on="category", how="left").sort_values(
        ["HIGH", "Total"], ascending=False)


def rule_catalogue() -> pd.DataFrame:
    """The full rule list -- printed into the working papers as documentation."""
    return pd.DataFrame([{
        "Code": r.code, "Check": r.title, "Severity": r.severity,
        "Area": r.category, "Basis / reference": r.reference,
    } for r in REGISTRY]).sort_values("Code")
