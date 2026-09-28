"""Debtors / creditors summary and bill-wise ageing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..normalize import r2


def bucket_labels(edges: list[int]) -> list[str]:
    labels, prev = [], 0
    for e in edges:
        labels.append(f"{prev}-{e}")
        prev = e + 1
    labels.append(f">{edges[-1]}")
    return labels


def _bucket(days: float, edges: list[int], labels: list[str]) -> str:
    if pd.isna(days):
        return "Undated"
    for i, e in enumerate(edges):
        if days <= e:
            return labels[i]
    return labels[-1]


def age_bills(bills: pd.DataFrame, cfg: Config, kind: str) -> dict[str, pd.DataFrame]:
    """Age one outstanding statement.

    Ageing runs off the **bill date** (what auditors report) and overdue days
    run off the **due date** (what management chases).
    """
    edges = [int(x) for x in cfg.th("ageing_buckets", [30, 60, 90, 180, 365])]
    labels = bucket_labels(edges)
    asof = cfg.period_to

    if bills is None or bills.empty:
        return {"detail": pd.DataFrame(), "summary": pd.DataFrame(),
                "party": pd.DataFrame(), "buckets": pd.DataFrame()}

    d = bills.copy()
    d["age_days"] = (asof - pd.to_datetime(d["bill_date"])).dt.days
    d["overdue_days"] = (asof - pd.to_datetime(d["due_date"])).dt.days.clip(lower=0)
    d["bucket"] = d["age_days"].map(lambda x: _bucket(x, edges, labels))
    d["overdue"] = d["overdue_days"] > 0
    d["as_on"] = asof

    # party-wise position
    party = (d.groupby("party", dropna=False)
               .agg(bills=("balance", "size"),
                    outstanding=("balance", "sum"),
                    oldest_bill=("bill_date", "min"),
                    max_age_days=("age_days", "max"),
                    overdue_amount=("balance", lambda s: r2(
                        d.loc[s.index][d.loc[s.index, "overdue"]]["balance"].sum())))
               .reset_index()
               .sort_values("outstanding", ascending=False))
    party["outstanding"] = party["outstanding"].round(2)
    total = party["outstanding"].sum()
    party["% of total"] = (100 * party["outstanding"] / total).round(2) if total else 0.0
    party["cumulative %"] = party["% of total"].cumsum().round(2)

    # bucket cross-tab, party x bucket
    pivot = (d.pivot_table(index="party", columns="bucket", values="balance",
                           aggfunc="sum", fill_value=0.0)
               .reindex(columns=[l for l in labels + ["Undated"]], fill_value=0.0)
               .round(2).reset_index())
    pivot["Total"] = pivot[[c for c in pivot.columns if c != "party"]].sum(axis=1).round(2)
    pivot = pivot.sort_values("Total", ascending=False)

    bucket_total = (d.groupby("bucket")["balance"].agg(["count", "sum"])
                     .reindex(labels + ["Undated"]).fillna(0).reset_index())
    bucket_total.columns = ["Ageing bucket", "Bills", "Amount"]
    bucket_total["Amount"] = bucket_total["Amount"].round(2)
    bucket_total["% "] = (100 * bucket_total["Amount"] / total).round(2) if total else 0.0

    summary = pd.DataFrame([{
        "Statement": kind.title(),
        "As on": asof.date(),
        "Parties": d["party"].nunique(),
        "Open bills": len(d),
        "Total outstanding": r2(total),
        "Overdue bills": int(d["overdue"].sum()),
        "Overdue amount": r2(d.loc[d["overdue"], "balance"].sum()),
        "Oldest bill": d["bill_date"].min(),
        "Max age (days)": int(np.nan_to_num(d["age_days"].max())),
        "Top-5 party concentration %": r2(party["% of total"].head(5).sum()),
    }])

    return {"detail": d.sort_values(["party", "bill_date"]),
            "summary": summary, "party": party, "buckets": bucket_total,
            "pivot": pivot}


def run(data: dict[str, pd.DataFrame], cfg: Config) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for kind in ("debtors", "creditors"):
        res = age_bills(data.get(kind, pd.DataFrame()), cfg, kind)
        for k, v in res.items():
            out[f"{kind}_{k}"] = v
    return out
