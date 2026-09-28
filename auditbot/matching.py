"""Generic record-matching engine.

Reconciliation is the same problem everywhere in an audit: two lists that
*should* agree, joined on keys that are never clean. This module does it once,
as a cascade of progressively looser passes, and every reconciliation module
reuses it.

Pass 1 might be "same party + same invoice number" (certain).
Pass 2 "same party + same amount within 7 days" (near certain).
Pass 3 "same amount within 7 days, party name 90%% similar" (probable).

Each match carries the pass name and a confidence score, so the reviewer can
see *why* two rows were tied together and override it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    _HAVE_RF = True
except ImportError:                                   # pragma: no cover
    import difflib
    _HAVE_RF = False


def similarity(a: str, b: str) -> float:
    """0-100 similarity between two normalised party names."""
    a, b = str(a or ""), str(b or "")
    if not a or not b:
        return 0.0
    if a == b:
        return 100.0
    if _HAVE_RF:
        return float(max(fuzz.token_set_ratio(a, b), fuzz.partial_ratio(a, b) * 0.95))
    return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


@dataclass
class Pass:
    """One rung of the matching cascade."""
    name: str
    keys: Sequence[tuple[str, str]] = field(default_factory=tuple)
    amount: Optional[tuple[str, str]] = None
    amount_tol: float = 1.0
    date: Optional[tuple[str, str]] = None
    date_tol: int = 7
    fuzzy: Optional[tuple[str, str]] = None
    fuzzy_threshold: float = 88.0
    sign: str = "same"          # "same" | "opposite" (for contra-style matching)
    confidence: float = 100.0   # ceiling for this pass


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([None] * len(df), index=df.index)


def _blocks(left: pd.DataFrame, right: pd.DataFrame, p: Pass
            ) -> list[tuple[pd.Index, pd.Index]]:
    """Cut the cartesian product down to comparable groups."""
    if p.keys:
        lk = pd.Series([""] * len(left), index=left.index)
        rk = pd.Series([""] * len(right), index=right.index)
        for lc, rc in p.keys:
            lk = lk + "|" + _series(left, lc).astype(str).str.upper().str.strip()
            rk = rk + "|" + _series(right, rc).astype(str).str.upper().str.strip()
        # a blank key component matches nothing -- avoids joining every row
        # that happens to have an empty invoice number
        bad_l = lk.str.contains(r"\|\s*$|\|\|", regex=True)
        bad_r = rk.str.contains(r"\|\s*$|\|\|", regex=True)
        lk, rk = lk[~bad_l], rk[~bad_r]
        groups = []
        rmap = rk.groupby(rk).groups
        for key, lidx in lk.groupby(lk).groups.items():
            if key in rmap:
                groups.append((pd.Index(lidx), pd.Index(rmap[key])))
        return groups

    if p.amount:
        lc, rc = p.amount
        la = _series(left, lc).astype(float).abs().round(0).astype("Int64")
        ra = _series(right, rc).astype(float).abs().round(0).astype("Int64")
        rmap: dict = {}
        for bucket, idx in ra.groupby(ra).groups.items():
            rmap.setdefault(int(bucket), []).extend(idx)
        span = int(np.ceil(p.amount_tol)) + 1
        groups = []
        for bucket, lidx in la.groupby(la).groups.items():
            cand: list = []
            for off in range(-span, span + 1):
                cand.extend(rmap.get(int(bucket) + off, []))
            if cand:
                groups.append((pd.Index(lidx), pd.Index(cand)))
        return groups

    return [(left.index, right.index)]


def _score_block(left: pd.DataFrame, right: pd.DataFrame, lidx: pd.Index,
                 ridx: pd.Index, p: Pass) -> list[tuple]:
    """Score every candidate pair inside one block."""
    out: list[tuple] = []
    lsub, rsub = left.loc[lidx], right.loc[ridx]

    lamt = lsub[p.amount[0]].astype(float).values if p.amount else None
    ramt = rsub[p.amount[1]].astype(float).values if p.amount else None
    ldt = pd.to_datetime(lsub[p.date[0]]).values if p.date else None
    rdt = pd.to_datetime(rsub[p.date[1]]).values if p.date else None
    lfz = lsub[p.fuzzy[0]].astype(str).values if p.fuzzy else None
    rfz = rsub[p.fuzzy[1]].astype(str).values if p.fuzzy else None

    for i, li in enumerate(lidx):
        for j, rj in enumerate(ridx):
            parts: list[float] = []
            if p.amount is not None:
                a, b = lamt[i], ramt[j]
                if p.sign == "opposite":
                    b = -b
                diff = abs(abs(a) - abs(b)) if p.sign == "abs" else abs(a - b)
                if diff > p.amount_tol:
                    continue
                parts.append(1.0 - diff / max(p.amount_tol, 1e-9))
            if p.date is not None:
                if pd.isna(ldt[i]) or pd.isna(rdt[j]):
                    continue
                days = abs((ldt[i] - rdt[j]) / np.timedelta64(1, "D"))
                if days > p.date_tol:
                    continue
                parts.append(1.0 - days / max(p.date_tol, 1e-9))
            if p.fuzzy is not None:
                sim = similarity(lfz[i], rfz[j])
                if sim < p.fuzzy_threshold:
                    continue
                parts.append(sim / 100.0)
            score = p.confidence * (sum(parts) / len(parts) if parts else 1.0)
            out.append((score, li, rj))
    return out


def cascade_match(left: pd.DataFrame, right: pd.DataFrame,
                  passes: Sequence[Pass],
                  left_label: str = "book", right_label: str = "source",
                  max_block: int = 400_000) -> dict[str, pd.DataFrame]:
    """Run the cascade. Returns ``{'matched', 'left_only', 'right_only'}``.

    Matching is strictly one-to-one: once a row is consumed by a pass it is
    withdrawn from all later passes.
    """
    left = left.copy()
    right = right.copy()
    if left.empty or right.empty:
        return {"matched": pd.DataFrame(), "left_only": left, "right_only": right}

    open_l, open_r = set(left.index), set(right.index)
    pairs: list[dict] = []

    for p in passes:
        if not open_l or not open_r:
            break
        lrem = left.loc[sorted(open_l)]
        rrem = right.loc[sorted(open_r)]
        scored: list[tuple] = []
        for lidx, ridx in _blocks(lrem, rrem, p):
            if len(lidx) * len(ridx) > max_block:
                # pathological block (e.g. hundreds of identical amounts):
                # fall back to positional pairing by date order
                lidx = pd.Index(sorted(lidx)[: int(np.sqrt(max_block))])
                ridx = pd.Index(sorted(ridx)[: int(np.sqrt(max_block))])
            scored.extend(_score_block(left, right, lidx, ridx, p))

        scored.sort(key=lambda t: -t[0])
        for score, li, rj in scored:
            if li in open_l and rj in open_r:
                open_l.discard(li)
                open_r.discard(rj)
                pairs.append({"left_idx": li, "right_idx": rj,
                              "match_pass": p.name, "confidence": round(score, 1)})

    matched = pd.DataFrame(pairs)
    if not matched.empty:
        lpart = left.loc[matched["left_idx"]].add_prefix(f"{left_label}_").reset_index(drop=True)
        rpart = right.loc[matched["right_idx"]].add_prefix(f"{right_label}_").reset_index(drop=True)
        matched = pd.concat(
            [matched[["match_pass", "confidence"]].reset_index(drop=True), lpart, rpart],
            axis=1)

    return {
        "matched": matched,
        "left_only": left.loc[sorted(open_l)].copy(),
        "right_only": right.loc[sorted(open_r)].copy(),
    }


# ------------------------------------------------------- bill-wise (FIFO)

def allocate_fifo(bills: pd.DataFrame, vouchers: pd.DataFrame,
                  party_col: str = "party_key",
                  bill_ref_col: str = "bill_key",
                  bill_no_col: str = "bill_key",
                  bill_date_col: str = "bill_date",
                  bill_amt_col: str = "bill_amount",
                  vch_amt_col: str = "amount",
                  vch_date_col: str = "txn_date") -> dict[str, pd.DataFrame]:
    """Bill-wise knock-off.

    Honours an explicit bill reference on the voucher first; anything left is
    applied oldest-bill-first within the same party. Returns the allocation
    lines, the bill-wise position, and vouchers that could not be applied
    (``on account``).
    """
    allocs: list[dict] = []
    if bills.empty or vouchers.empty:
        return {"allocations": pd.DataFrame(),
                "bill_status": bills.copy(),
                "unapplied": vouchers.copy()}

    b = bills.copy().reset_index(drop=True)
    v = vouchers.copy().reset_index(drop=True)
    b["_open"] = b[bill_amt_col].astype(float)
    v["_left"] = v[vch_amt_col].astype(float)

    # index bills by party for quick lookup
    by_party: dict[str, list[int]] = {}
    for i, key in enumerate(b[party_col].astype(str)):
        by_party.setdefault(key, []).append(i)

    def _apply(vi: int, bi: int, basis: str) -> None:
        amt = round(min(v.at[vi, "_left"], b.at[bi, "_open"]), 2)
        if amt <= 0:
            return
        v.at[vi, "_left"] = round(v.at[vi, "_left"] - amt, 2)
        b.at[bi, "_open"] = round(b.at[bi, "_open"] - amt, 2)
        allocs.append({
            "voucher_row": v.at[vi, "row_id"] if "row_id" in v.columns else vi,
            "voucher_no": v.at[vi, "voucher_no"] if "voucher_no" in v.columns else "",
            "voucher_date": v.at[vi, vch_date_col],
            "party": v.at[vi, "party"] if "party" in v.columns else "",
            "bill_row": b.at[bi, "row_id"] if "row_id" in b.columns else bi,
            "bill_no": b.at[bi, "bill_no"] if "bill_no" in b.columns else "",
            "bill_date": b.at[bi, bill_date_col],
            "bill_amount": b.at[bi, bill_amt_col],
            "allocated": amt,
            "basis": basis,
            "days_taken": (pd.Timestamp(v.at[vi, vch_date_col])
                           - pd.Timestamp(b.at[bi, bill_date_col])).days
            if pd.notna(v.at[vi, vch_date_col]) and pd.notna(b.at[bi, bill_date_col]) else None,
        })

    # -- pass 1: explicit bill reference on the voucher
    bill_lookup: dict[tuple[str, str], list[int]] = {}
    for i, (pk, bk) in enumerate(zip(b[party_col].astype(str), b[bill_no_col].astype(str))):
        if bk:
            bill_lookup.setdefault((pk, bk), []).append(i)
    for vi in v.index:
        ref = str(v.at[vi, bill_ref_col]) if bill_ref_col in v.columns else ""
        if not ref or v.at[vi, "_left"] <= 0:
            continue
        for bi in bill_lookup.get((str(v.at[vi, party_col]), ref), []):
            if v.at[vi, "_left"] <= 0:
                break
            _apply(vi, bi, "bill reference")

    # -- pass 2: FIFO within party, oldest bill first
    for party, idxs in by_party.items():
        idxs = sorted(idxs, key=lambda i: (pd.Timestamp(b.at[i, bill_date_col])
                                           if pd.notna(b.at[i, bill_date_col])
                                           else pd.Timestamp.max))
        vidx = v.index[v[party_col].astype(str) == party]
        vidx = sorted(vidx, key=lambda i: (pd.Timestamp(v.at[i, vch_date_col])
                                           if pd.notna(v.at[i, vch_date_col])
                                           else pd.Timestamp.max))
        for vi in vidx:
            for bi in idxs:
                if v.at[vi, "_left"] <= 0:
                    break
                if b.at[bi, "_open"] > 0:
                    _apply(vi, bi, "FIFO")

    bill_status = b.rename(columns={"_open": "unpaid_amount"})
    bill_status["settled_amount"] = (bill_status[bill_amt_col].astype(float)
                                     - bill_status["unpaid_amount"]).round(2)
    bill_status["status"] = np.where(
        bill_status["unpaid_amount"] <= 0.01, "SETTLED",
        np.where(bill_status["settled_amount"] > 0.01, "PART PAID", "OPEN"))

    unapplied = v[v["_left"] > 0.01].rename(columns={"_left": "unapplied_amount"})
    return {"allocations": pd.DataFrame(allocs),
            "bill_status": bill_status,
            "unapplied": unapplied}


def find_duplicates(df: pd.DataFrame, subset: Sequence[str],
                    amount_col: str | None = None,
                    date_col: str | None = None,
                    date_window: int = 0) -> pd.DataFrame:
    """Rows that repeat on *subset*. With *date_window* > 0 the dates need only
    be close, which is how real duplicate payments look."""
    if df.empty or not set(subset).issubset(df.columns):
        return pd.DataFrame()
    work = df.copy()
    for c in subset:
        work[c] = work[c].astype(str).str.upper().str.strip()
    if amount_col:
        work["_amt"] = work[amount_col].astype(float).round(2)
        subset = list(subset) + ["_amt"]

    dups = work[work.duplicated(subset=list(subset), keep=False)].copy()
    if dups.empty or not date_col or date_window <= 0:
        if not dups.empty:
            dups["dup_group"] = dups.groupby(list(subset), dropna=False).ngroup() + 1
        return dups.drop(columns=["_amt"], errors="ignore")

    keep: list[pd.DataFrame] = []
    for _, grp in dups.groupby(list(subset), dropna=False):
        grp = grp.sort_values(date_col)
        dates = pd.to_datetime(grp[date_col])
        if dates.isna().all() or (dates.max() - dates.min()).days <= date_window:
            keep.append(grp)
    out = pd.concat(keep) if keep else pd.DataFrame()
    if not out.empty:
        out["dup_group"] = out.groupby(list(subset), dropna=False).ngroup() + 1
    return out.drop(columns=["_amt"], errors="ignore")
