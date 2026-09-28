"""Purchase and sales reconciliation.

Ties each register to the sub-ledger it feeds: sales -> debtors -> receipts,
purchases -> creditors -> payments. A party-wise control account is built for
both sides, plus month trends and sales invoice sequence testing.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..config import Config
from ..normalize import r2


def _control(register: pd.DataFrame, vouchers: pd.DataFrame,
             outstanding: pd.DataFrame, side: str) -> pd.DataFrame:
    """Party-wise control account: billed, settled, closing balance per books."""
    frames = []
    if register is not None and not register.empty:
        frames.append(register.groupby(["party_key", "party"])["total"]
                      .sum().round(2).rename("billed"))
    if vouchers is not None and not vouchers.empty:
        frames.append(vouchers.groupby(["party_key", "party"])["amount"]
                      .sum().round(2).rename("settled"))
    if outstanding is not None and not outstanding.empty:
        frames.append(outstanding.groupby(["party_key", "party"])["balance"]
                      .sum().round(2).rename("closing_per_ledger"))
    if not frames:
        return pd.DataFrame()

    m = pd.concat(frames, axis=1).fillna(0.0).reset_index()
    m["derived_closing"] = (m["billed"] - m["settled"]).round(2) \
        if "billed" in m and "settled" in m else 0.0
    m["difference"] = (m["closing_per_ledger"] - m["derived_closing"]).round(2) \
        if "closing_per_ledger" in m else 0.0
    m["side"] = side
    total = m["billed"].sum() if "billed" in m else 0.0
    m["% of turnover"] = (100 * m["billed"] / total).round(2) if total else 0.0
    return m.sort_values("billed", ascending=False)


def _monthly(register: pd.DataFrame, vouchers: pd.DataFrame, label: str) -> pd.DataFrame:
    if register is None or register.empty:
        return pd.DataFrame()
    reg = (register.groupby("month")
                   .agg(invoices=("total", "size"), taxable=("taxable", "sum"),
                        tax=("tax", "sum"), value=("total", "sum"))
                   .round(2).reset_index())
    if vouchers is not None and not vouchers.empty:
        v = vouchers.groupby("month")["amount"].sum().round(2).rename(f"{label}_settled")
        reg = reg.merge(v, left_on="month", right_index=True, how="outer").fillna(0.0)
    reg = reg.sort_values("month")
    reg["value_mom %"] = (100 * reg["value"].pct_change()).round(2)
    reg["effective_tax %"] = np.where(reg["taxable"] > 0,
                                      (100 * reg["tax"] / reg["taxable"]).round(2), 0.0)
    return reg


_SEQ = re.compile(r"(\d+)\s*$")


def invoice_sequence(sales: pd.DataFrame) -> pd.DataFrame:
    """Gaps and repeats in the outward invoice series.

    Series is the invoice number with its trailing digits stripped, so
    ``INV/25-26/001`` and ``INV/25-26/002`` are compared with each other and
    not with ``EXP/25-26/001``.
    """
    if sales is None or sales.empty:
        return pd.DataFrame()
    rows = []
    work = sales.copy()
    work["_num"] = work["invoice_no"].astype(str).str.extract(_SEQ)[0]
    work = work[work["_num"].notna()]
    if work.empty:
        return pd.DataFrame()
    work["_num"] = work["_num"].astype(int)
    work["_series"] = (work["invoice_no"].astype(str)
                       .str.replace(_SEQ, "", regex=True).str.strip())

    for series, grp in work.groupby("_series"):
        nums = sorted(grp["_num"].tolist())
        lo, hi = nums[0], nums[-1]
        missing = sorted(set(range(lo, hi + 1)) - set(nums))
        repeated = sorted({n for n in nums if nums.count(n) > 1})
        rows.append({
            "series": series or "(numeric only)",
            "from": lo, "to": hi, "invoices": len(nums),
            "expected": hi - lo + 1,
            "missing_count": len(missing),
            "missing_numbers": ", ".join(map(str, missing[:40]))
                               + (" ..." if len(missing) > 40 else ""),
            "repeated_numbers": ", ".join(map(str, repeated[:20])),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("missing_count", ascending=False)


def run(data: dict[str, pd.DataFrame], cfg: Config) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    sales_ctrl = _control(data.get("sales"), data.get("receipts"),
                          data.get("debtors"), "Sales / Debtors")
    purch_ctrl = _control(data.get("purchase"), data.get("payments"),
                          data.get("creditors"), "Purchase / Creditors")
    if not sales_ctrl.empty:
        out["sales_control"] = sales_ctrl
    if not purch_ctrl.empty:
        out["purchase_control"] = purch_ctrl

    sm = _monthly(data.get("sales"), data.get("receipts"), "receipts")
    pm = _monthly(data.get("purchase"), data.get("payments"), "payments")
    if not sm.empty:
        out["sales_trend"] = sm
    if not pm.empty:
        out["purchase_trend"] = pm

    seq = invoice_sequence(data.get("sales"))
    if not seq.empty:
        out["invoice_sequence"] = seq

    # headline reconciliation card
    def tot(df, col):
        return r2(df[col].sum()) if df is not None and not df.empty and col in df else 0.0

    out["trade_summary"] = pd.DataFrame([
        {"Head": "Sales (register)", "Value": tot(data.get("sales"), "total")},
        {"Head": "Receipts (vouchers)", "Value": tot(data.get("receipts"), "amount")},
        {"Head": "Debtors closing (ledger)", "Value": tot(data.get("debtors"), "balance")},
        {"Head": "Purchases (register)", "Value": tot(data.get("purchase"), "total")},
        {"Head": "Payments (vouchers)", "Value": tot(data.get("payments"), "amount")},
        {"Head": "Creditors closing (ledger)", "Value": tot(data.get("creditors"), "balance")},
    ])
    return out
