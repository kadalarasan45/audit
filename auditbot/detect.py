"""Work out what a file *is*, without being told.

Drop any mix of client files in any folder, named anything, in CSV / Excel /
PDF. This module reads each one's headers, scores them against every alias
section in ``column_maps.yaml``, and decides which of the fifteen sources it
is -- so nothing depends on a filename or a folder layout.

Some pairs are structurally identical (a receipts register and a payments
register have the same columns; so do debtors and creditors outstandings).
Those are resolved in a second pass using the filename and, more reliably, by
checking whose parties the file contains -- customers or suppliers.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import Config
from .io_utils import _alias_index, find_header_row, read_raw, slug
from .normalize import norm_party

log = logging.getLogger("auditbot.detect")

# every source the pipeline knows, and the alias section it is read with
SOURCE_SECTION = {
    "bank": "bank", "bankbook": "bank", "cashbook": "bank",
    "purchase": "purchase", "sales": "sales", "gstr2b": "gstr2b",
    "gstr1": "gst_summary", "gstr3b": "gst_summary",
    "receipts": "receipts", "payments": "payments",
    "debtors": "outstanding", "creditors": "outstanding",
    "salary": "salary", "inventory": "inventory", "conversion": "conversion",
}

LABEL = {
    "bank": "Bank statement", "bankbook": "Bank ledger (books)",
    "cashbook": "Cash book", "purchase": "Purchase register",
    "sales": "Sales register", "gstr2b": "GSTR-2B (portal)",
    "gstr1": "GSTR-1 summary", "gstr3b": "GSTR-3B summary",
    "receipts": "Receipts register", "payments": "Payments register",
    "debtors": "Debtors outstanding", "creditors": "Creditors outstanding",
    "salary": "Salary register", "inventory": "Stock summary",
    "conversion": "Production / conversion",
}

# filename words that hint at a source. Only ever a tie-breaker.
NAME_HINTS = {
    "bank": ["statement", "stmt", "passbook", "hdfc", "icici", "sbi", "axis",
             "kotak", "yesbank", "idbi", "canara", "unionbank", "indusind",
             "bob", "pnb", "federal", "cub", "karur", "tmb"],
    "bankbook": ["bankbook", "bankledger", "bankaccount", "banka/c", "bookbank"],
    "cashbook": ["cashbook", "cashledger", "pettycash", "cashaccount"],
    "purchase": ["purchase", "inward", "vendorinvoice", "supplierinvoice", "pi"],
    "sales": ["sales", "outward", "customerinvoice", "salesinvoice", "si"],
    "gstr2b": ["gstr2b", "gstr2a", "2b", "2a", "autodrafted", "itcstatement"],
    "gstr1": ["gstr1", "r1", "outwardreturn"],
    "gstr3b": ["gstr3b", "3b", "returnsummary", "taxliability"],
    "receipts": ["receipt", "collection", "moneyreceived", "rcpt"],
    "payments": ["payment", "paid", "disbursement", "pmt"],
    "debtors": ["debtor", "receivable", "sundrydebtor", "billsreceivable",
                "customeroutstanding", "ar"],
    "creditors": ["creditor", "payable", "sundrycreditor", "billspayable",
                  "supplieroutstanding", "ap"],
    "salary": ["salary", "payroll", "wage", "paysheet", "pay", "ctc", "muster"],
    "inventory": ["stock", "inventory", "closingstock", "stocksummary", "godown"],
    "conversion": ["conversion", "production", "manufactur", "jobwork", "bom",
                   "consumption", "mfg"],
}

# headers that, if present, all but settle it
STRONG_SIGNALS = {
    "gstr2b": ["gstinofsupplier", "itcavailability", "gstr1gstr5filingdate",
               "gstr1iffgstr5period", "tradelegalname", "gstr1gstr1a5period"],
    "gstr3b": ["itccgst", "itcigst", "paidincash", "paidbyitc", "challanno"],
    "salary": ["uan", "esi", "pf", "netpay", "grosssalary", "professionaltax"],
    "inventory": ["closingqty", "openingqty", "stockgroup", "closingvalue"],
    "conversion": ["standardyield", "finisheditem", "rawmaterial", "outputqty"],
    "purchase": ["suppliername", "supplierinvoiceno", "vendorname", "itcavailability"],
    "sales": ["customername", "buyername", "placeofsupply"],
    "receipts": ["receiptno", "receiptdate", "receivedfrom", "modeofreceipt"],
    "payments": ["paymentno", "paymentdate", "paidto", "modeofpayment"],
    "debtors": ["receivable"],
    "creditors": ["payable"],
    "bank": ["withdrawalamt", "depositamt", "closingbalance", "chqrefno",
             "valuedate", "narration"],
}

def _has_ledger_columns(f: set, sig: dict) -> bool:
    """A ledger or statement has separate debit and credit columns; a voucher
    register has one signed amount.

    This has to be judged on the raw headers, not on the fields as the section
    maps them -- inside the receipts section a column called "Credit" is
    already an alias for *amount*, so a bank ledger would look like a receipts
    register to its own gate.
    """
    return sig["dr_cr"] or ("amount" in f and "drcr" in f)


# Hard structural requirements. A source that fails its gate is not a
# candidate at all, however well its columns happen to score -- this is what
# stops a bank ledger being read as a receipts register.
GATES = {
    "bank":       _has_ledger_columns,
    "bankbook":   _has_ledger_columns,
    "cashbook":   _has_ledger_columns,
    "receipts":   lambda f, s: "amount" in f and not _has_ledger_columns(f, s),
    "payments":   lambda f, s: "amount" in f and not _has_ledger_columns(f, s),
    "purchase":   lambda f, s: "invoice_no" in f and bool({"taxable", "total"} & f),
    "sales":      lambda f, s: "invoice_no" in f and bool({"taxable", "total"} & f),
    "gstr2b":     lambda f, s: "invoice_no" in f and bool({"gstin", "party"} & f),
    "gstr1":      lambda f, s: "period" in f and "invoice_no" not in f,
    "gstr3b":     lambda f, s: ("period" in f and "invoice_no" not in f
                                and bool({"itc_cgst", "itc_sgst", "itc_igst",
                                          "paid_cash", "paid_credit"} & f)),
    "debtors":    lambda f, s: bool({"balance", "bill_amount"} & f
                                    and {"bill_no", "bill_date"} & f),
    "creditors":  lambda f, s: bool({"balance", "bill_amount"} & f
                                    and {"bill_no", "bill_date"} & f),
    "salary":     lambda f, s: bool({"net", "gross", "basic"} & f),
    "inventory":  lambda f, s: bool({"closing_qty", "opening_qty", "closing_val"} & f),
    "conversion": lambda f, s: bool({"output_qty", "input_qty", "output_item"} & f),
}

# sources that share a column layout and need the second pass
AMBIGUOUS = [("receipts", "payments"), ("debtors", "creditors"),
             ("bank", "bankbook", "cashbook"), ("gstr1", "gstr3b"),
             ("purchase", "sales")]


@dataclass
class Detection:
    path: Path
    source: str | None
    confidence: float
    rows: int
    headers: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    frame: pd.DataFrame | None = None
    sheet: str = ""

    @property
    def label(self) -> str:
        return LABEL.get(self.source or "", "Not recognised")

    @property
    def why(self) -> str:
        return "; ".join(self.reasons)


def _name_key(path: Path) -> str:
    return re.sub(r"[^a-z0-9]", "", path.stem.lower())


def _fields(headers: list[str], section: str, cfg: Config) -> set[str]:
    """Canonical field names the headers resolve to, for one alias section."""
    idx = _alias_index(cfg.aliases(section))
    return {idx[h] for h in headers if h in idx}


def _score_section(headers: list[str], section: str, cfg: Config) -> float:
    """Fraction of the section's canonical fields present in the headers."""
    aliases = cfg.aliases(section)
    if not aliases:
        return 0.0
    return len(_fields(headers, section, cfg)) / len(aliases)


def _best_sheet(path: Path, cfg: Config) -> tuple[str, list[str], pd.DataFrame] | None:
    """Pick the sheet with the most recognisable headers, across all sections."""
    try:
        sheets = read_raw(path, cfg.aliases("bank"))
    except Exception as exc:
        log.warning("could not read %s: %s", path.name, exc)
        return None

    best = None
    best_hits = -1
    for name, raw in sheets:
        if raw is None or raw.empty:
            continue
        for section in set(SOURCE_SECTION.values()):
            hr = find_header_row(raw, cfg.aliases(section))
            headers = [slug(c) for c in raw.iloc[hr].tolist() if slug(c)]
            idx = _alias_index(cfg.aliases(section))
            hits = sum(1 for h in headers if h in idx)
            if hits > best_hits:
                best_hits, best = hits, (name, headers, raw)
    return best if best_hits > 1 else None


def detect_file(path: Path, cfg: Config) -> Detection:
    """Classify one file. Structure first, filename only as a tie-breaker."""
    found = _best_sheet(path, cfg)
    if found is None:
        return Detection(path, None, 0.0, 0, reasons=["no recognisable table found"])
    sheet, headers, raw = found
    hset = set(headers)
    name = _name_key(path)

    bank_fields = _fields(headers, "bank", cfg)
    signals = {"dr_cr": {"debit", "credit"} <= bank_fields,
               "balance": "balance" in bank_fields}

    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    for source, section in SOURCE_SECTION.items():
        fields = _fields(headers, section, cfg)
        gate = GATES.get(source)
        if gate and not gate(fields, signals):
            scores[source] = 0.0
            reasons[source] = []
            continue
        score = _score_section(headers, section, cfg)
        why = []
        strong = [s for s in STRONG_SIGNALS.get(source, []) if s in hset]
        if strong:
            score += 0.28 * min(len(strong), 3)
            why.append(f"columns {', '.join(strong[:3])}")
        hits = [h for h in NAME_HINTS.get(source, []) if h in name]
        if hits:
            score += 0.18
            why.append(f"filename says '{hits[0]}'")
        scores[source] = score
        reasons[source] = why

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, top_score = ranked[0]
    if top_score < 0.30:
        return Detection(path, None, round(top_score, 2), len(raw) - 1, headers,
                         ["headers did not match any known document"],
                         [(s, round(v, 2)) for s, v in ranked[:3]], raw, sheet)

    why = reasons[top] or [f"column layout matches a {LABEL[top].lower()}"]
    return Detection(path, top, round(min(top_score, 1.0), 2), len(raw) - 1,
                     headers, why, [(s, round(v, 2)) for s, v in ranked[1:4]],
                     raw, sheet)


# ------------------------------------------------- second pass: who's who

def _party_column(det: Detection, cfg: Config) -> set[str]:
    """Normalised party names in a detected file, for the overlap test."""
    if det.frame is None or det.source is None:
        return set()
    section = SOURCE_SECTION.get(det.source, det.source)
    from .io_utils import map_columns
    try:
        mapped = map_columns(det.frame, cfg.aliases(section))
    except Exception:
        return set()
    for col in ("party", "description"):
        if col in mapped.columns:
            return {norm_party(v) for v in mapped[col].dropna().head(400)} - {""}
    return set()


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(len(a), 1) if a and b else 0.0


def resolve(dets: list[Detection], cfg: Config) -> list[Detection]:
    """Separate the look-alike pairs using party overlap.

    A receipts register and a payments register have identical columns. What
    tells them apart is *whose* names are in it: customers or suppliers. The
    sales and purchase registers, already identified, provide both lists.
    """
    by_source: dict[str, Detection] = {}
    for d in dets:
        if d.source and d.source not in by_source:
            by_source[d.source] = d

    customers = _party_column(by_source["sales"], cfg) if "sales" in by_source else set()
    suppliers = _party_column(by_source["purchase"], cfg) if "purchase" in by_source else set()
    if not customers and not suppliers:
        return dets

    for d in dets:
        if d.source not in {"receipts", "payments", "debtors", "creditors"}:
            continue
        parties = _party_column(d, cfg)
        if not parties:
            continue
        c, s = _overlap(parties, customers), _overlap(parties, suppliers)
        if abs(c - s) < 0.15:            # no clear winner, leave the guess alone
            continue
        pair = ("receipts", "payments") if d.source in ("receipts", "payments") \
            else ("debtors", "creditors")
        chosen = pair[0] if c > s else pair[1]
        if chosen != d.source:
            d.reasons.append(
                f"reclassified from {LABEL[d.source].lower()}: "
                f"{max(c, s):.0%} of parties are "
                f"{'customers' if c > s else 'suppliers'}")
            d.source = chosen
            d.confidence = round(min(d.confidence + 0.1, 1.0), 2)
        else:
            d.reasons.append(
                f"{max(c, s):.0%} of parties are "
                f"{'customers' if c > s else 'suppliers'}")
    return dets


_BANKISH = re.compile(
    r"neft|rtgs|imps|upi|chq|cheque|atm|ecs|nach|ach|clg|inw|outw|"
    r"charge|int\.|interest|emi|pos |mandate|refund", re.I)
_CASHISH = re.compile(r"\bcash\b|petty|self|withdraw|cash dep|csw|cdm", re.I)


def _descriptions(det: Detection, cfg: Config) -> pd.Series:
    if det.frame is None:
        return pd.Series(dtype=str)
    from .io_utils import map_columns
    try:
        m = map_columns(det.frame, cfg.aliases("bank"))
    except Exception:
        return pd.Series(dtype=str)
    return m["description"].dropna().astype(str) if "description" in m.columns \
        else pd.Series(dtype=str)


def _resolve_bank_family(dets: list[Detection], cfg: Config) -> None:
    """Separate a bank statement, a bank ledger and a cash book.

    All three are date/particulars/debit/credit, so the columns cannot tell
    them apart. What can: a cash book's narrations are about cash, and a
    statement carries the bank's own running balance where a ledger export
    usually does not.
    """
    family = [d for d in dets if d.source in {"bank", "bankbook", "cashbook"}]
    if len(family) < 2:
        return

    profiled = []
    for d in family:
        desc = _descriptions(d, cfg)
        cash_share = float(desc.str.contains(_CASHISH).mean()) if len(desc) else 0.0
        bank_share = float(desc.str.contains(_BANKISH).mean()) if len(desc) else 0.0
        has_balance = bool(
            d.frame is not None
            and "balance" in _fields(d.headers, "bank", cfg))
        profiled.append((d, cash_share, bank_share, has_balance))

    for d, cash_share, bank_share, has_balance in profiled:
        if cash_share >= 0.5 and cash_share > bank_share:
            new, why = "cashbook", f"{cash_share:.0%} of narrations are about cash"
        elif has_balance:
            new, why = "bank", "carries the bank's own running balance"
        else:
            new, why = "bankbook", "ledger layout with no running balance column"
        if new != d.source:
            d.reasons.append(f"reclassified from {LABEL[d.source].lower()}: {why}")
            d.source = new
        else:
            d.reasons.append(why)
        # these three are genuinely hard to tell apart; say so honestly
        d.confidence = round(min(d.confidence, 0.85), 2)


def _dedupe(dets: list[Detection]) -> list[Detection]:
    """Two files claiming the same single-file source: keep the confident one,
    unless the source legitimately takes many files (bank statements do)."""
    multi = {"bank"}
    seen: dict[str, Detection] = {}
    for d in sorted(dets, key=lambda x: -x.confidence):
        if not d.source or d.source in multi:
            continue
        if d.source in seen:
            loser = d.source
            d.alternatives = [(loser, d.confidence)] + d.alternatives
            # fall back to the best alternative that is still free, rather
            # than throwing the file away
            nxt = next(((s2, v) for s2, v in d.alternatives[1:]
                        if v > 0.25 and (s2 in multi or s2 not in seen)), None)
            if nxt:
                d.reasons.append(
                    f"{seen[loser].path.name} is the better "
                    f"{LABEL[loser].lower()}, so read as a {LABEL[nxt[0]].lower()}")
                d.source, d.confidence = nxt[0], round(nxt[1], 2)
                if d.source not in multi:
                    seen[d.source] = d
            else:
                d.reasons.append(
                    f"also looked like a {LABEL[loser].lower()}, but "
                    f"{seen[loser].path.name} matched better")
                d.source, d.confidence = None, 0.0
        else:
            seen[d.source] = d
    return dets


def scan(folder: Path | str, cfg: Config,
         patterns: tuple[str, ...] = ("*.csv", "*.xlsx", "*.xlsm", "*.xls",
                                      "*.pdf", "*.txt", "*.tsv")) -> list[Detection]:
    """Classify every readable file under *folder*, recursively."""
    folder = Path(folder)
    files: list[Path] = []
    for pat in patterns:
        files.extend(p for p in folder.rglob(pat)
                     if p.is_file() and not p.name.startswith(("~$", ".")))
    dets = [detect_file(p, cfg) for p in sorted(set(files))]
    dets = resolve(dets, cfg)
    _resolve_bank_family(dets, cfg)
    return _dedupe(dets)


def to_sources(dets: list[Detection]) -> dict[str, list[str]]:
    """Turn detections into the ``sources:`` mapping the pipeline expects."""
    out: dict[str, list[str]] = {}
    for d in dets:
        if d.source:
            out.setdefault(d.source, []).append(str(d.path))
    return out


def summary_frame(dets: list[Detection]) -> pd.DataFrame:
    return pd.DataFrame([{
        "File": d.path.name,
        "Detected as": d.label,
        "Confidence": f"{d.confidence:.0%}",
        "Rows": d.rows,
        "Why": d.why,
    } for d in dets])
