# Audit Office Automation

Turns a month-end pile of client files — bank statements, Tally exports, GST
portal downloads, payroll and stock registers — into reconciled working papers
and a ranked exception register, in about a second per client.

```
raw files ─▶ ingest ─▶ classify ─▶ match ─▶ reconcile ─▶ 51 audit checks ─▶ reports
 csv/xlsx/pdf          narration    cascade   BRS, ageing,   exception        Excel + HTML
 any layout            + party      matcher   GST, payroll   register         + Power BI
```

Nothing is sent anywhere. It runs entirely on the machine the client data is
already on.

---

## Quick start

```bash
./run.sh serve      # opens a page you drag files onto  <- start here
```

`run.sh` builds the virtual environment on first use. The other commands:

```bash
./run.sh sample                  # generate a realistic demo client (FY 2025-26)
./run.sh detect data/inbox       # say what each file is, and stop
./run.sh run --inbox data/inbox  # auto-detect everything, then reconcile
./run.sh run                     # use the paths configured in config.yaml
./run.sh check                   # show which files the config resolves to
./run.sh rules                   # list every check the system performs
```

---

## Feeding it files

**Nothing depends on filenames or folders.** Drop whatever you have, however it
is named, wherever you like. The system reads each file's *column headings* and
works out what it is; the filename is only ever a tie-breaker.

| Constraint | Reality |
|---|---|
| Format | CSV, Excel (.xlsx/.xls/.xlsm), PDF, tab-separated text |
| Filename | irrelevant — `doc_4.xlsx` works as well as `sales_register.xlsx` |
| Folder layout | irrelevant — subfolders are scanned recursively |
| Completeness | send a subset; missing documents are skipped, not fatal |
| Order | irrelevant |

Three ways in, all equivalent:

1. **`./run.sh serve`** — a local page at `http://127.0.0.1:8765`. Drag files
   on, it shows what it thinks each one is with a confidence and a reason, you
   correct anything wrong in a dropdown, then it runs and hands back the
   dashboard, the workbook and the CSVs. Bound to loopback; nothing leaves the
   machine.
2. **`./run.sh run --inbox <folder>`** — same detection, from the terminal.
   `data/inbox/` exists for this.
3. **`./run.sh run`** — explicit paths in `config.yaml`, for a scheduled job
   where the files always land in the same place.

### How it tells the documents apart

Column headings alone settle most of it. Three pairs are structurally
identical, so they need more:

- **Receipts vs payments**, **debtors vs creditors** — same columns. Resolved
  by checking *whose* names are in the file: customers (from the sales
  register) or suppliers (from the purchase register).
- **Bank statement vs bank ledger vs cash book** — all date/particulars/debit/
  credit. Resolved by the running balance column (a statement carries the
  bank's own) and by how many narrations are about cash.

Tested blind: with every file renamed to `doc_1.xlsx`, `file_12.xlsx` and so on
and scattered across subfolders, it identifies **15 of 15** and produces
byte-identical reconciliation results. There is a test that enforces this.

When it cannot place a file it says so and prints the headings it found, rather
than guessing. Add those headings to `config/column_maps.yaml` and it is
recognised next time — no code change.

Outputs land in `output/`:

| File | What it is |
|---|---|
| `Audit_Working_Papers.xlsx` | 54 indexed, filterable sheets — the file you review and file |
| `dashboard.html` | one-page partner view; opens in any browser, works offline |
| `exception_register.csv` | every finding, flat, for a query letter mail-merge |
| `powerbi/` | star-schema CSVs + a model guide for Power BI |

---

## The twelve activities, and where each one lands

| # | Activity | Module | Output |
|---|---|---|---|
| 1 | Bank statement import & reconciliation | `recon/bank.py` | BRS, *Bank Not In Books*, *Cheques Uncleared* |
| 2 | Debtors summary & ageing | `recon/ageing.py` | *Debtors Summary / Party / Ageing / Bills* |
| 3 | Creditors summary & ageing | `recon/ageing.py` | *Creditors Summary / Party / Ageing / Bills* |
| 4 | Bill-wise receipts | `recon/billwise.py` | *Billwise Receipts*, *Receipts On Account*, *Collection Days* |
| 5 | Bill-wise payments | `recon/billwise.py` | *Billwise Payments*, *Payments On Account*, *Payment Days* |
| 6 | CGST / SGST payment tracking | `recon/gst.py` | *GST Payment*, *Books vs GSTR-3B*, *Books vs GSTR-1* |
| 7 | Cash expenses above ₹10,000 | `exceptions.py` (CSH-01..05) | *Cash Above Limit* |
| 8 | Salary expense processing | `recon/salary.py` | *Salary Summary / Monthly / Employee / Bank Tieup* |
| 9 | Inventory conversion | `recon/inventory.py` | *Conversion Summary / Detail / Item* |
| 10 | Purchase reconciliation | `recon/gst.py`, `recon/trade.py` | *ITC Not In 2B*, *2B Not In Books*, *Purchase Control* |
| 11 | Sales reconciliation | `recon/gst.py`, `recon/trade.py` | *Sales Control*, *Sales Trend*, *Invoice Sequence* |
| 12 | Automated audit exception reporting | `exceptions.py` | *Exceptions*, *Exception Summary*, dashboard |

---

## Setting it up for a real client

### 1. Point it at the files

Everything lives in `config/config.yaml`. Drop the client's files into
`data/input/` (or anywhere — the paths are globs) and edit:

```yaml
client:
  name: "Acme Industries Private Limited"
  gstin: "33AABCA1234A1Z5"
  state_code: "33"
  fy: "2025-26"
  period_from: "2025-04-01"
  period_to: "2026-03-31"

sources:
  bank:     ["clients/acme/bank/*.pdf", "clients/acme/bank/*.csv"]
  purchase: ["clients/acme/purchase_register.xlsx"]
  # any source you leave out is simply skipped
```

Run `./run.sh check` to confirm each source resolves before running.

### 2. Tune the thresholds

Every number a check uses is in `thresholds:` — the ₹10,000 s.40A(3) limit,
the ₹2,00,000 s.269ST limit, ageing buckets, the 180-day Rule 37 window, the
fuzzy-match cut-off, PF/ESI ceilings, TDS trigger points. A trading client and
a manufacturer use the same code with different numbers.

### 3. Teach it a new column heading

Column headings are matched against `config/column_maps.yaml`, ignoring case,
spaces and punctuation. When a client's export uses a heading nobody has seen
before, add the alias — **no Python changes**:

```yaml
bank:
  debit: [debit, withdrawal, withdrawalamt, dr, paidout]   # <- added "paidout"
```

Unrecognised columns are never dropped; they come through prefixed `x_` and
appear in the working papers so you can see what was ignored.

### 4. Add a check

```python
@rule("CSH-06", "Cash paid to a related party", "HIGH", "Cash", "s.40A(2)(b)")
def related_party_cash(data, recon, cfg):
    pay = data["payments"]
    hits = pay[pay["is_cash"] & pay["party"].isin(cfg.get("related_parties", []))]
    return _mk(_rows(hits, dataset="payments", date="txn_date", party="party",
                     amount="amount",
                     observation=lambda r: f"Cash of Rs.{inr(r['amount'])} to a related party",
                     action="Verify commercial justification under s.40A(2)(b)."))
```

That is the whole change — the register, the dashboard, the Excel sheet and the
rule catalogue all pick it up.

---

## How the matching works

Reconciliation is one engine (`matching.py`) used everywhere, run as a cascade
of progressively looser passes. Each match records *which* pass caught it and a
confidence score, so a reviewer can see why two rows were tied together:

| Pass | Rule | Confidence |
|---|---|---|
| 1 | GSTIN + invoice number | 100 |
| 2 | GSTIN + value + date within tolerance | 96 |
| 3 | party + invoice number | 94 |
| 4 | party + value + date | 90 |
| 5 | fuzzy party name + value + date | 82 |
| 6 | value + date only | 65 |

Matching is strictly one-to-one — a row consumed by an earlier pass is withdrawn
from all later ones, so nothing is double-counted.

Bill-wise knock-off (`allocate_fifo`) honours an explicit bill reference on the
voucher first, then applies the remainder oldest-bill-first within the same
party. Whatever cannot be applied is reported as *on account* rather than
silently absorbed.

---

## What the checks cover

51 checks across nine areas. `./run.sh rules` prints the catalogue with the
statutory basis for each.

- **Cash** — s.40A(3) single and same-day aggregate, s.269ST receipts, negative cash
- **Duplicates** — purchase and sales invoices, payments, receipts, bank lines
- **GST** — ITC not in GSTR-2B (s.16(2)(aa)), 2B not booked, per-invoice value drift, GSTR-3B short payment, books vs GSTR-1/3B, GSTIN validity and place-of-supply tax heads, cash paid despite unused ITC, missing challan
- **Bank** — statement entries not in books, stale cheques, BRS that does not tie, unbooked charges and interest
- **Debtors / creditors** — over 180 days, past the credit period, credit balances in debtor accounts, Rule 37 ITC reversal on suppliers unpaid 180 days, concentration
- **Bill-wise** — receipts and payments left on account
- **Salary** — net pay arithmetic, PF and ESI against statutory ceilings, cash salary, shared bank account / PAN / UAN (ghost employees), missing PAN, payroll-to-bank tie-up, month-on-month swings
- **Inventory** — negative stock, closing versus movement, conversion yield variance, slow moving, valuation rate drift
- **Analytical** — round sums, year-end cut-off, money moved on holidays, missing narration, invoice-series gaps, TDS thresholds, parties with no GSTIN

Findings are ranked HIGH → MEDIUM → LOW, then by value involved, so the first
page of the register is the day's work.

---

## Layout

```
auditbot/
  config.py        configuration + dotted lookups
  normalize.py     amounts, dates, party names, GSTIN checksum
  io_utils.py      CSV/Excel/PDF readers, header detection, column mapping
  ingest.py        typed canonical datasets + narration classification
  matching.py      the cascade matcher, FIFO allocation, duplicate finder
  exceptions.py    the 51 checks and the register builder
  pipeline.py      orchestration
  recon/           bank, ageing, billwise, gst, trade, salary, inventory
  report_excel.py  the working-papers workbook
  charts.py        inline-SVG chart primitives
  dashboard.py     the one-page HTML dashboard
  powerbi.py       star-schema CSV exports
  detect.py        works out what each file is, from its column headings
  server.py        the local drag-and-drop tester
  cli.py           command line
config/            config.yaml + column_maps.yaml  ← the only files you edit
data/inbox/        drop client files here, any format, any name
scripts/           sample data generator
tests/             58 tests: python -m unittest discover -s tests
```

---

## Where it fits with Tally and Power Automate

- **Tally** — export Day Book, bill-wise outstandings, the bank and cash ledgers
  and the stock summary to Excel. The aliases already cover Tally's headings.
  Nothing here writes back into Tally; posting the exceptions stays a human act.
- **Power Automate Desktop** — use it for the parts that are portal clicking
  (downloading GSTR-2B/1/3B, filing acknowledgements) and drop the files into
  the client's `data/input/`. The reconciliation itself is far cheaper to do
  here than as recorded clicks.
- **Power BI** — point it at `output/powerbi/` and refresh. The folder ships a
  model guide with the relationships and starter DAX.

---

## Assumptions worth stating

- Dates are read **day-first** (05-04-2025 is 5 April), the Indian convention.
- Bank statement sign convention: debit = withdrawal. Ledger exports are the
  mirror image, and a brought-forward line is treated as opening balance, not
  as a transaction to be matched.
- GSTIN is validated for format *and* the mod-36 check digit; it is not verified
  against the portal.
- Statutory limits in `config.yaml` reflect FY 2025-26. **Confirm them against
  the current Finance Act before relying on a run** — they are configuration,
  deliberately, so they are one edit to correct.
- Everything the system produces is an *exception to be examined*, not a
  conclusion. The suggested action on each finding is a starting point for the
  auditor's judgement, not advice.
- No client data is in this repository and none should ever be committed. The
  demo dataset is generated on demand by `scripts/make_sample_data.py`;
  spreadsheets and PDFs are blanket-ignored by extension so none can reach a
  commit by accident.

---

## Copyright

Copyright (c) 2026 Kadalarasan. All rights reserved.

This source is published for reference. It carries **no licence**, so no
permission is granted to use, copy, modify or distribute it. If you would like
to use it, please ask.
