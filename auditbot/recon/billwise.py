"""Bill-wise receipts and payments.

Applies receipt/payment vouchers against open bills (explicit reference first,
then FIFO), and reports collection/payment performance per party.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..matching import allocate_fifo
from ..normalize import r2


def _perform(alloc: pd.DataFrame, label: str) -> pd.DataFrame:
    if alloc.empty:
        return pd.DataFrame()
    g = (alloc.groupby("party", dropna=False)
              .agg(**{f"{label}_count": ("allocated", "size"),
                      f"{label}_amount": ("allocated", "sum"),
                      "avg_days": ("days_taken", "mean"),
                      "max_days": ("days_taken", "max")})
              .reset_index())
    g[f"{label}_amount"] = g[f"{label}_amount"].round(2)
    g["avg_days"] = g["avg_days"].round(1)
    return g.sort_values(f"{label}_amount", ascending=False)


def bill_population(register: pd.DataFrame, outstanding: pd.DataFrame,
                    default_days: int) -> pd.DataFrame:
    """The set of bills a voucher can be knocked off against.

    The invoice register is the right population -- the outstanding statement
    only lists what is *still* unpaid, so allocating against it would leave
    almost every receipt looking like an on-account payment. Bills that exist
    only in the outstanding statement (opening balances, advances) are added
    on top.
    """
    frames = []
    if register is not None and not register.empty:
        frames.append(pd.DataFrame({
            "row_id": register["row_id"],
            "party": register["party"], "party_key": register["party_key"],
            "bill_no": register["invoice_no"], "bill_key": register["inv_key"],
            "bill_date": register["txn_date"],
            "bill_amount": register["total"].astype(float),
            "origin": "register",
        }))
    if outstanding is not None and not outstanding.empty:
        known = set(frames[0]["bill_key"]) if frames else set()
        extra = outstanding[~outstanding["bill_key"].isin(known)]
        if not extra.empty:
            frames.append(pd.DataFrame({
                "row_id": extra["row_id"],
                "party": extra["party"], "party_key": extra["party_key"],
                "bill_no": extra["bill_no"], "bill_key": extra["bill_key"],
                "bill_date": extra["bill_date"],
                "bill_amount": extra["bill_amount"].astype(float),
                "origin": "opening / advance",
            }))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out[out["bill_amount"] > 0].reset_index(drop=True)


def _side(bills: pd.DataFrame, vouchers: pd.DataFrame, cfg: Config,
          label: str) -> dict[str, pd.DataFrame]:
    if bills is None or bills.empty or vouchers is None or vouchers.empty:
        return {f"{label}_allocations": pd.DataFrame(),
                f"{label}_bill_status": pd.DataFrame(),
                f"{label}_unapplied": pd.DataFrame(),
                f"{label}_performance": pd.DataFrame(),
                f"{label}_summary": pd.DataFrame()}

    res = allocate_fifo(bills, vouchers)
    alloc, status, unapplied = res["allocations"], res["bill_status"], res["unapplied"]

    if not alloc.empty:
        alloc["allocated"] = alloc["allocated"].round(2)
        alloc["beyond_credit_period"] = alloc["days_taken"].fillna(0) > int(
            cfg.th(f"{'debtor' if label == 'receipts' else 'creditor'}_credit_days_default", 45))

    total_vch = r2(vouchers["amount"].sum())
    applied = r2(alloc["allocated"].sum()) if not alloc.empty else 0.0
    summary = pd.DataFrame([{
        "Side": label.title(),
        "Vouchers": len(vouchers),
        "Voucher value": total_vch,
        "Applied against bills": applied,
        "On account / unapplied": r2(total_vch - applied),
        "Bills fully settled": int((status["status"] == "SETTLED").sum()),
        "Bills part paid": int((status["status"] == "PART PAID").sum()),
        "Bills untouched": int((status["status"] == "OPEN").sum()),
        "Matched by bill reference %": r2(
            100 * (alloc["basis"] == "bill reference").mean()) if not alloc.empty else 0.0,
    }])

    return {f"{label}_allocations": alloc,
            f"{label}_bill_status": status,
            f"{label}_unapplied": unapplied,
            f"{label}_performance": _perform(alloc, label),
            f"{label}_summary": summary}


def run(data: dict[str, pd.DataFrame], cfg: Config) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    sales_bills = bill_population(data.get("sales"), data.get("debtors"),
                                  int(cfg.th("debtor_credit_days_default", 45)))
    purch_bills = bill_population(data.get("purchase"), data.get("creditors"),
                                  int(cfg.th("creditor_credit_days_default", 45)))
    out.update(_side(sales_bills, data.get("receipts"), cfg, "receipts"))
    out.update(_side(purch_bills, data.get("payments"), cfg, "payments"))
    return out
