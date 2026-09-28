"""Cleaning primitives: amounts, dates, party names, invoice numbers, GSTIN."""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------- amounts

_CURRENCY = "₹$"
_AMT_STRIP = re.compile(r"[,\s" + _CURRENCY + r"]")
_TRAILING_DRCR = re.compile(r"\b(dr|cr|debit|credit)\.?$", re.I)


def clean_amount(value) -> float:
    """Parse anything an accountant can type into a float.

    Handles ``1,23,456.78``, ``(1234)`` negatives, ``Rs. 500``, ``1200 Dr``,
    ``1200 Cr`` (Cr is returned positive, Dr negative -- callers that want raw
    magnitudes should use :func:`abs`), blanks, dashes and NaN.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return 0.0 if pd.isna(value) else float(value)

    s = str(value).strip()
    if not s or s in {"-", "--", "nil", "NIL", "N/A", "na", "NA", "None"}:
        return 0.0

    sign = 1.0
    if s.startswith("(") and s.endswith(")"):
        sign, s = -1.0, s[1:-1]

    m = _TRAILING_DRCR.search(s)
    if m:
        if m.group(1).lower() in {"dr", "debit"}:
            sign = -sign
        s = s[: m.start()].strip()

    s = _AMT_STRIP.sub("", s)
    if s.startswith("-"):
        sign, s = -sign, s[1:]
    if not s:
        return 0.0
    try:
        return sign * float(s)
    except ValueError:
        # last resort: pull the first numeric run out of the string
        m = re.search(r"\d+(?:\.\d+)?", s)
        return sign * float(m.group()) if m else 0.0


def r2(value) -> float:
    """Round to 2 dp, the only rounding used in reports."""
    try:
        return round(float(value) + 0.0, 2)
    except (TypeError, ValueError):
        return 0.0


# ------------------------------------------------------------------ dates

_DATE_FORMATS = (
    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%y", "%d/%m/%y",
    "%Y-%m-%d", "%Y/%m/%d", "%d-%b-%Y", "%d-%b-%y", "%d %b %Y",
    "%d-%B-%Y", "%b %d, %Y", "%m/%d/%Y", "%Y%m%d", "%d%m%Y",
)


def parse_date(value) -> Optional[pd.Timestamp]:
    """Parse a date, preferring Indian day-first order."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()
    if isinstance(value, date):
        return pd.Timestamp(value)
    if isinstance(value, (int, float)):
        # Excel serial date (1900 epoch, as written by Excel itself)
        if 20000 < float(value) < 60000:
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(value))
        return None

    s = str(value).strip()
    if not s or s.lower() in {"nan", "nat", "none", "-"}:
        return None
    s = re.sub(r"\s+", " ", s.split(" 00:00")[0])
    for fmt in _DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt)).normalize()
        except ValueError:
            continue
    try:
        ts = pd.to_datetime(s, dayfirst=True, errors="coerce")
        return None if pd.isna(ts) else pd.Timestamp(ts).normalize()
    except Exception:
        return None


def parse_period(value) -> Optional[str]:
    """Normalise a return period to ``YYYY-MM`` (accepts ``042025``, ``Apr-25``)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if re.fullmatch(r"\d{6}", s):          # GST portal MMYYYY
        return f"{s[2:]}-{s[:2]}"
    ts = parse_date(s if re.search(r"\d{4}", s) else f"01-{s}")
    return ts.strftime("%Y-%m") if ts is not None else None


# ----------------------------------------------------------- text / names

_LEGAL_SUFFIX = re.compile(
    r"\b(private|pvt|limited|ltd|llp|inc|corp|corporation|company|co|"
    r"and|&|sons|bros|brothers|enterprises|enterprise|traders|trading|"
    r"industries|india|services|solutions|the)\b",
    re.I,
)
_NON_ALNUM = re.compile(r"[^A-Z0-9 ]")
_ACCOUNT_NOISE = re.compile(
    r"\b(a/c|ac|account|dr|cr|opening|closing|balance|sundry|debtors|creditors)\b", re.I
)


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def norm_party(value) -> str:
    """Aggressive party-name key used for fuzzy matching and grouping.

    ``M/s. Sharma & Sons Pvt. Ltd.`` and ``SHARMA AND SONS PRIVATE LIMITED``
    both collapse to ``SHARMA``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = _ascii(str(value)).upper()
    s = re.sub(r"^M/?S\.?\s+", "", s)
    s = _NON_ALNUM.sub(" ", s)
    s = _ACCOUNT_NOISE.sub(" ", s)
    s = _LEGAL_SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def display_party(value) -> str:
    """Tidy but human-readable party name kept on the report face."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", _ascii(str(value))).strip()


def norm_invoice(value) -> str:
    """Invoice-number key: uppercase alphanumerics only, leading zeros dropped.

    ``INV/2025-26/0042`` -> ``INV20252642``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = re.sub(r"[^A-Z0-9]", "", _ascii(str(value)).upper())
    return s.lstrip("0") or s


def invoice_tail(value, n: int = 6) -> str:
    """Last *n* significant characters -- survives prefix differences between
    the supplier's invoice number in books and on the GST portal."""
    s = norm_invoice(value)
    return s[-n:] if len(s) > n else s


# ------------------------------------------------------------------ GSTIN

GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}[Z][0-9A-Z]{1}$")
PAN_RE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
_GST_CODES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def clean_gstin(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def gstin_checksum_ok(gstin: str) -> bool:
    """Validate the 15th character (mod-36 weighted checksum)."""
    if len(gstin) != 15:
        return False
    total = 0
    for i, ch in enumerate(gstin[:14]):
        if ch not in _GST_CODES:
            return False
        val = _GST_CODES.index(ch) * (2 if i % 2 else 1)
        total += val // 36 + val % 36
    return _GST_CODES[(36 - total % 36) % 36] == gstin[14]


def validate_gstin(value) -> tuple[bool, str]:
    """Return ``(is_valid, reason)``."""
    g = clean_gstin(value)
    if not g:
        return False, "GSTIN missing"
    if len(g) != 15:
        return False, f"GSTIN length {len(g)} (expected 15)"
    if not GSTIN_RE.match(g):
        return False, "GSTIN format invalid"
    if not gstin_checksum_ok(g):
        return False, "GSTIN check digit invalid"
    return True, ""


def gstin_state(value) -> str:
    g = clean_gstin(value)
    return g[:2] if len(g) >= 2 else ""


def pan_from_gstin(value) -> str:
    g = clean_gstin(value)
    return g[2:12] if len(g) == 15 else ""


def validate_pan(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(PAN_RE.match(re.sub(r"[^A-Z0-9]", "", str(value).upper())))


# ------------------------------------------------------------- misc flags

def is_round_sum(amount: float, multiple: float) -> bool:
    """True for suspiciously round figures (1,00,000 / 5,00,000 ...)."""
    a = abs(float(amount or 0))
    return a >= multiple and a % multiple == 0


def fy_bounds(fy: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``2025-26`` -> (2025-04-01, 2026-03-31)."""
    start_year = int(str(fy).split("-")[0])
    return pd.Timestamp(start_year, 4, 1), pd.Timestamp(start_year + 1, 3, 31)


def inr(value) -> str:
    """Format in the Indian lakh/crore grouping: 12,34,567.89."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return ""
    sign = "-" if n < 0 else ""
    whole, _, frac = f"{abs(n):.2f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        whole = f"{head},{tail}"
    return f"{sign}{whole}.{frac}"
