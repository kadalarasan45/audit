"""File readers + automatic column mapping.

The whole point of this module: an audit office receives the same data in a
hundred different layouts. Nothing downstream should ever care. Everything
here funnels CSV / XLSX / PDF into a DataFrame whose columns are the canonical
names declared in ``config/column_maps.yaml``.
"""
from __future__ import annotations

import glob
import io
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger("auditbot.io")

_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def slug(text) -> str:
    """Header key: lowercase alphanumerics only. ``Withdrawal Amt.`` -> ``withdrawalamt``."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(text).strip().lower())


def expand(patterns: Iterable[str], root: Path) -> list[Path]:
    """Resolve globs relative to the project root, skipping Excel lock files."""
    out: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        hits = glob.glob(str(p if p.is_absolute() else root / p))
        out.extend(Path(h) for h in sorted(hits))
    return [p for p in out if p.is_file() and not p.name.startswith(("~$", "."))]


# --------------------------------------------------------------- readers

def _sniff_sep(text: str) -> str:
    """Pick the delimiter by majority vote across lines.

    Sniffing off the first line is unreliable -- bank exports start with a
    banner ("HDFC BANK LIMITED") that has no delimiter at all, and pandas'
    own sniffer then settles on whitespace and shreds the file.
    """
    lines = [l for l in text.splitlines() if l.strip()][:200]
    if not lines:
        return ","
    best, best_score = ",", -1.0
    for sep in (",", ";", "\t", "|"):
        counts = [l.count(sep) for l in lines]
        if not counts:
            continue
        common = max(set(counts), key=counts.count)
        if common == 0:
            continue
        # how many lines agree on that field count, weighted by width
        score = counts.count(common) / len(counts) * common
        if score > best_score:
            best, best_score = sep, score
    return best


def _read_csv(path: Path) -> list[tuple[str, pd.DataFrame]]:
    last: Exception | None = None
    for enc in _ENCODINGS:
        try:
            text = path.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError) as exc:
            last = exc
            continue
        sep = _sniff_sep(text)
        # Fix the column count up front: a banner row with no delimiter would
        # otherwise make pandas size the frame at one column and then discard
        # every real data row as "bad".
        width = max((l.count(sep) for l in text.splitlines()), default=0) + 1
        try:
            df = pd.read_csv(io.StringIO(text), header=None, dtype=str, sep=sep,
                             engine="python", skip_blank_lines=False,
                             names=list(range(width)), on_bad_lines="skip")
            return [(path.stem, df)]
        except Exception as exc:                       # pragma: no cover
            last = exc
    raise RuntimeError(f"cannot read {path}: {last}")


def _read_excel(path: Path) -> list[tuple[str, pd.DataFrame]]:
    engine = "openpyxl" if path.suffix.lower() in {".xlsx", ".xlsm"} else None
    book = pd.read_excel(path, sheet_name=None, header=None, dtype=object, engine=engine)
    return [(name, df) for name, df in book.items() if not df.empty]


_PDF_ROW = re.compile(
    r"^(?P<date>\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\s+"
    r"(?P<desc>.+?)\s+"
    r"(?P<nums>(?:[\d,]+\.\d{2}\s*){1,3})$"
)
_DATE_START = re.compile(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}")


def _pdf_lines(page, tol: float = 2.0) -> list[list[dict]]:
    """Group a page's words into visual lines, left to right."""
    lines: dict[int, list[dict]] = {}
    for w in page.extract_words(use_text_flow=False, keep_blank_chars=False):
        key = int(round(float(w["top"]) / tol))
        lines.setdefault(key, []).append(w)
    return [sorted(ws, key=lambda w: float(w["x0"]))
            for _, ws in sorted(lines.items())]


def _column_edges(header_words: list[dict], gap: float = 12.0) -> list[float]:
    """Turn a header line into column boundaries.

    Header cells are multi-word ("Withdrawal Amt."), so words are first grouped
    on the horizontal gap between them, then the boundary between two columns
    is put midway between the groups.
    """
    groups: list[list[dict]] = []
    for w in header_words:
        if groups and float(w["x0"]) - float(groups[-1][-1]["x1"]) <= gap:
            groups[-1].append(w)
        else:
            groups.append([w])
    edges = [0.0]
    for a, b in zip(groups, groups[1:]):
        edges.append((float(a[-1]["x1"]) + float(b[0]["x0"])) / 2)
    edges.append(float("inf"))
    return edges


def _pdf_by_position(pdf, aliases: dict[str, list[str]]) -> pd.DataFrame | None:
    """Rebuild a statement printed without grid lines, using x-positions.

    Packing the trailing numbers left-to-right (the naive approach) puts the
    credit under the debit heading whenever a row has no debit -- which is most
    rows. Assigning each word to the column its x-position falls in keeps blank
    cells blank.
    """
    idx = _alias_index(aliases)
    header: list[dict] | None = None
    edges: list[float] = []
    rows: list[list[str]] = []

    for page in pdf.pages:
        for words in _pdf_lines(page):
            if header is None:
                hits = sum(1 for w in words if slug(w["text"]) in idx)
                if hits >= 3:
                    header = words
                    edges = _column_edges(words)
                    rows.append([" ".join(w["text"] for w in words
                                          if edges[i] <= (float(w["x0"]) + float(w["x1"])) / 2
                                          < edges[i + 1])
                                 for i in range(len(edges) - 1)])
                continue
            if not words or not _DATE_START.match(words[0]["text"]):
                continue
            cells = []
            for i in range(len(edges) - 1):
                cells.append(" ".join(
                    w["text"] for w in words
                    if edges[i] <= (float(w["x0"]) + float(w["x1"])) / 2 < edges[i + 1]))
            rows.append(cells)

    if header is None or len(rows) < 2:
        return None
    return pd.DataFrame(rows)


def _read_pdf(path: Path, aliases: dict[str, list[str]] | None = None
              ) -> list[tuple[str, pd.DataFrame]]:
    """Bank-statement PDFs: ruled tables first, then an x-position rebuild for
    statements printed without grid lines, then a plain line parser."""
    try:
        import pdfplumber
    except ImportError:                                # pragma: no cover
        log.warning("pdfplumber not installed - skipping %s", path.name)
        return []

    tables: list[pd.DataFrame] = []
    lines: list[list[str]] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            for tbl in page.extract_tables() or []:
                if len(tbl) > 1 and len(tbl[0]) > 2:
                    tables.append(pd.DataFrame(tbl))
        if tables:
            return [(f"{path.stem}_t{i}", t) for i, t in enumerate(tables)]

        if aliases:
            positioned = _pdf_by_position(pdf, aliases)
            if positioned is not None:
                return [(path.stem, positioned)]

        for page in pdf.pages:
            for raw in (page.extract_text() or "").splitlines():
                m = _PDF_ROW.match(raw.strip())
                if not m:
                    continue
                nums = m.group("nums").split()
                row = [m.group("date"), m.group("desc").strip()] + nums
                lines.append(row + [""] * (5 - len(row)))

    if lines:
        width = max(len(r) for r in lines)
        header = (["Date", "Narration", "Debit", "Credit", "Balance"] + [""] * width)[:width]
        return [(path.stem, pd.DataFrame([header] + [r[:width] for r in lines]))]
    log.warning("no tabular content found in %s", path.name)
    return []


def read_raw(path: Path, aliases: dict[str, list[str]] | None = None
             ) -> list[tuple[str, pd.DataFrame]]:
    """Return ``[(sheet_name, headerless DataFrame)]`` for any supported file.

    *aliases* is only needed for PDFs, where knowing the expected headings is
    what lets the reader find the header line and the column boundaries.
    """
    ext = path.suffix.lower()
    if ext in {".csv", ".txt", ".tsv"}:
        return _read_csv(path)
    if ext in {".xlsx", ".xlsm", ".xls"}:
        return _read_excel(path)
    if ext == ".pdf":
        return _read_pdf(path, aliases)
    log.warning("unsupported file type: %s", path.name)
    return []


# ------------------------------------------------------- header handling

def _alias_index(aliases: dict[str, list[str]]) -> dict[str, str]:
    """``{slugged alias -> canonical}``. Canonical names map to themselves."""
    idx: dict[str, str] = {}
    for canon, alts in aliases.items():
        idx.setdefault(slug(canon), canon)
        for alt in alts or []:
            idx.setdefault(slug(alt), canon)
    return idx


def find_header_row(raw: pd.DataFrame, aliases: dict[str, list[str]],
                    scan: int = 40) -> int:
    """Pick the row that looks most like a header.

    Score = number of cells that resolve to a canonical field. Tie-breaks on
    how many cells are non-empty, so a banner row of merged cells never wins.
    """
    idx = _alias_index(aliases)
    best, best_score, best_fill = 0, -1, -1
    for i in range(min(scan, len(raw))):
        row = raw.iloc[i]
        hits = sum(1 for cell in row if slug(cell) in idx)
        fill = int(row.notna().sum())
        if hits > best_score or (hits == best_score and hits > 0 and fill > best_fill):
            best, best_score, best_fill = i, hits, fill
    return best if best_score > 0 else 0


def map_columns(raw: pd.DataFrame, aliases: dict[str, list[str]],
                header_row: int | None = None) -> pd.DataFrame:
    """Slice off the header row and rename columns to canonical names.

    Unrecognised columns are kept with an ``x_`` prefix so nothing is silently
    lost -- they show up in the working papers for manual review.
    """
    if raw.empty:
        return pd.DataFrame()
    idx = _alias_index(aliases)
    hr = find_header_row(raw, aliases) if header_row is None else header_row

    header = [str(c) if c is not None else "" for c in raw.iloc[hr].tolist()]
    body = raw.iloc[hr + 1:].reset_index(drop=True)
    body.columns = range(len(header))

    names, seen = [], set()
    for pos, cell in enumerate(header):
        canon = idx.get(slug(cell))
        if canon is None:
            canon = f"x_{slug(cell)}" if slug(cell) else f"x_col{pos}"
        while canon in seen:                      # duplicate headers happen
            canon = f"{canon}_2"
        seen.add(canon)
        names.append(canon)
    body.columns = names

    body = body.dropna(how="all")
    # drop rows that are clearly totals/footers
    first = body.columns[0]
    mask = body[first].astype(str).str.strip().str.lower().isin(
        {"total", "grand total", "totals", "sub total", "subtotal"})
    return body[~mask].reset_index(drop=True)


def load_frames(patterns: Iterable[str], section: str, cfg,
                root: Path) -> pd.DataFrame:
    """Read every file matching *patterns*, map to canonical columns for
    *section*, and stack them with provenance columns."""
    aliases = cfg.aliases(section)
    files = expand(patterns, root)
    parts: list[pd.DataFrame] = []
    for path in files:
        try:
            sheets = read_raw(path, aliases)
        except Exception as exc:
            log.error("failed reading %s: %s", path.name, exc)
            continue
        for sheet_name, raw in sheets:
            mapped = map_columns(raw, aliases)
            if mapped.empty:
                continue
            canon_hits = sum(1 for c in mapped.columns if not c.startswith("x_"))
            if canon_hits < 2:
                log.debug("skipping %s!%s (only %d recognised columns)",
                          path.name, sheet_name, canon_hits)
                continue
            mapped["source_file"] = path.name
            mapped["source_sheet"] = str(sheet_name)
            parts.append(mapped)
    if not parts:
        log.info("no data loaded for section '%s'", section)
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    log.info("loaded %-11s %5d rows from %d file(s)", section, len(out), len(files))
    return out
