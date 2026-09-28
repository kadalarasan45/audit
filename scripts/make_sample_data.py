#!/usr/bin/env python3
"""Generate a realistic FY 2025-26 sample client dataset.

The data is deliberately imperfect: invoices missing from GSTR-2B, cash
payments over the 40A(3) limit, a duplicate payment, stale cheques, negative
stock, a ghost-employee bank account and so on -- so a full pipeline run has
something to find. Deterministic (seeded), so reruns give identical files.

    python scripts/make_sample_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from auditbot.normalize import _GST_CODES  # noqa: E402

IN = ROOT / "data" / "input"
RNG = np.random.default_rng(20260328)
START, END = pd.Timestamp("2025-04-01"), pd.Timestamp("2026-03-31")
MONTHS = pd.date_range(START, END, freq="MS")


def gstin(state: str, pan: str, entity: str = "1") -> str:
    base = f"{state}{pan}{entity}Z"
    total = 0
    for i, ch in enumerate(base):
        v = _GST_CODES.index(ch) * (2 if i % 2 else 1)
        total += v // 36 + v % 36
    return base + _GST_CODES[(36 - total % 36) % 36]


def pan(seed: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    r = np.random.default_rng(seed)
    return ("".join(r.choice(list(letters), 3)) + "PA"
            + f"{r.integers(1000, 9999)}" + r.choice(list(letters)))


CUSTOMERS = [
    ("Velan Agencies", "33"), ("Kumaran Steel Traders", "33"),
    ("Anand Hardware Stores", "33"), ("Sri Balaji Enterprises", "33"),
    ("Deccan Auto Parts Pvt Ltd", "29"), ("Meridian Distributors LLP", "27"),
    ("Sharma & Sons", "33"), ("Nova Retail India Pvt Ltd", "36"),
    ("Coastal Trading Co", "33"), ("Prime Industrial Supplies", "33"),
    ("Everest Marketing", "07"), ("Sunrise Traders", "33"),
]
SUPPLIERS = [
    ("Tirupur Textiles Pvt Ltd", "33"), ("Global Polymers Ltd", "27"),
    ("Madras Packaging Works", "33"), ("Neelkanth Chemicals", "24"),
    ("Sathya Transport Service", "33"), ("Unity Engineering Works", "33"),
    ("Bharat Fasteners Pvt Ltd", "29"), ("Lakshmi Power Solutions", "33"),
    ("Orient Logistics LLP", "33"), ("Precision Tools India", "33"),
    ("Metro Office Supplies", "33"), ("Skyline Advertising", "33"),
]


def parties(names, tag):
    rows = []
    for i, (nm, st) in enumerate(names):
        p = pan(hash((tag, i)) % 10_000_000)
        rows.append({"name": nm, "state": st, "gstin": gstin(st, p),
                     "credit_days": int(RNG.choice([15, 30, 45, 60]))})
    return pd.DataFrame(rows)


CUST = parties(CUSTOMERS, "C")
SUPP = parties(SUPPLIERS, "S")


def rnd_date(month: pd.Timestamp) -> pd.Timestamp:
    last = (month + pd.offsets.MonthEnd(0)).day
    return month + pd.Timedelta(days=int(RNG.integers(0, last)))


def gst_split(taxable: float, state: str, rate: float = 18.0) -> dict:
    tax = round(taxable * rate / 100, 2)
    if state == "33":
        return {"cgst": round(tax / 2, 2), "sgst": round(tax / 2, 2), "igst": 0.0}
    return {"cgst": 0.0, "sgst": 0.0, "igst": tax}


# ------------------------------------------------------------ registers

def build_sales() -> pd.DataFrame:
    rows, n = [], 0
    for mi, m in enumerate(MONTHS):
        for _ in range(int(RNG.integers(16, 24))):
            n += 1
            c = CUST.iloc[int(RNG.integers(0, len(CUST)))]
            taxable = float(np.round(RNG.uniform(18_000, 4_20_000), -1))
            t = gst_split(taxable, c["state"])
            rows.append({
                "Date": rnd_date(m).strftime("%d-%m-%Y"),
                "Invoice No": f"SI/25-26/{n:04d}",
                "Party Name": c["name"], "GSTIN": c["gstin"],
                "Taxable Value": taxable, **{k.upper(): v for k, v in t.items()},
                "Total Invoice Value": round(taxable + sum(t.values()), 2),
                "HSN": int(RNG.choice([72061010, 39269099, 84213990, 73181500])),
                "Place of Supply": c["state"], "Voucher No": f"SAL-{n:04d}",
            })
    df = pd.DataFrame(rows)
    # deliberate defects -------------------------------------------------
    df.loc[5, "Invoice No"] = df.loc[4, "Invoice No"]          # duplicate number
    df.loc[40, "GSTIN"] = "33AABCS1429B1ZZ"                    # bad check digit
    df = df.drop(index=[77, 78, 79]).reset_index(drop=True)    # sequence gaps
    df.loc[12, "Total Invoice Value"] = 5_00_000.0             # round sum
    return df


def build_purchase() -> pd.DataFrame:
    rows, n = [], 0
    for m in MONTHS:
        for _ in range(int(RNG.integers(12, 18))):
            n += 1
            s = SUPP.iloc[int(RNG.integers(0, len(SUPP)))]
            taxable = float(np.round(RNG.uniform(9_000, 3_10_000), -1))
            t = gst_split(taxable, s["state"])
            rows.append({
                "Date": rnd_date(m).strftime("%d-%m-%Y"),
                "Supplier Invoice No": f"{s['name'][:3].upper()}/{n:04d}",
                "Supplier Name": s["name"], "GSTIN": s["gstin"],
                "Taxable Value": taxable, **{k.upper(): v for k, v in t.items()},
                "Invoice Value": round(taxable + sum(t.values()), 2),
                "HSN": int(RNG.choice([52051110, 39011010, 48191010, 29051100])),
                "Place of Supply": s["state"], "Voucher No": f"PUR-{n:04d}",
                "ITC Availability": "Yes",
            })
    df = pd.DataFrame(rows)
    df.loc[9, ["Supplier Invoice No", "Supplier Name", "GSTIN"]] = \
        df.loc[8, ["Supplier Invoice No", "Supplier Name", "GSTIN"]].values  # duplicate
    df.loc[30, "GSTIN"] = ""                                    # missing GSTIN
    return df


def build_gstr2b(purchase: pd.DataFrame) -> pd.DataFrame:
    """Portal view: a few of the client's invoices never got filed by the
    supplier, a few extra ones the client never booked, some value drift."""
    keep = purchase.drop(index=list(range(0, len(purchase), 29))).copy()   # ~6 missing
    out = pd.DataFrame({
        "GSTIN of supplier": keep["GSTIN"],
        "Trade/Legal name": keep["Supplier Name"],
        "Invoice number": keep["Supplier Invoice No"],
        "Invoice Date": keep["Date"],
        "Invoice Value(Rs)": keep["Invoice Value"],
        "Taxable Value (Rs)": keep["Taxable Value"],
        "Integrated Tax(Rs)": keep["IGST"],
        "Central Tax(Rs)": keep["CGST"],
        "State/UT Tax(Rs)": keep["SGST"],
        "Cess(Rs)": 0.0,
        "GSTR-1/IFF/GSTR-5 Period": pd.to_datetime(keep["Date"], dayfirst=True).dt.strftime("%m%Y"),
        "ITC Availability": "Yes",
    }).reset_index(drop=True)

    for i in (3, 17, 44, 71, 98):                       # value drift
        if i < len(out):
            out.loc[i, "Taxable Value (Rs)"] = round(out.loc[i, "Taxable Value (Rs)"] * 0.97, 2)
            out.loc[i, "Central Tax(Rs)"] = round(out.loc[i, "Central Tax(Rs)"] * 0.97, 2)
            out.loc[i, "State/UT Tax(Rs)"] = round(out.loc[i, "State/UT Tax(Rs)"] * 0.97, 2)

    extra = []                                          # in 2B, never booked
    for k in range(4):
        s = SUPP.iloc[k]
        taxable = float(np.round(RNG.uniform(20_000, 1_50_000), -1))
        t = gst_split(taxable, s["state"])
        d = rnd_date(MONTHS[k + 2])
        extra.append({
            "GSTIN of supplier": s["gstin"], "Trade/Legal name": s["name"],
            "Invoice number": f"{s['name'][:3].upper()}/X{k:03d}",
            "Invoice Date": d.strftime("%d-%m-%Y"),
            "Invoice Value(Rs)": round(taxable + sum(t.values()), 2),
            "Taxable Value (Rs)": taxable, "Integrated Tax(Rs)": t["igst"],
            "Central Tax(Rs)": t["cgst"], "State/UT Tax(Rs)": t["sgst"], "Cess(Rs)": 0.0,
            "GSTR-1/IFF/GSTR-5 Period": d.strftime("%m%Y"), "ITC Availability": "Yes",
        })
    return pd.concat([out, pd.DataFrame(extra)], ignore_index=True)


def build_gst_returns(sales: pd.DataFrame, purchase: pd.DataFrame):
    s = sales.copy()
    s["period"] = pd.to_datetime(s["Date"], dayfirst=True).dt.strftime("%m%Y")
    g1 = (s.groupby("period")
            .agg(taxable=("Taxable Value", "sum"), cgst=("CGST", "sum"),
                 sgst=("SGST", "sum"), igst=("IGST", "sum"))
            .round(2).reset_index())
    g1.loc[2, "taxable"] = round(g1.loc[2, "taxable"] * 0.94, 2)     # under-reported
    g1.loc[7, "cgst"] = round(g1.loc[7, "cgst"] + 12_500, 2)         # over-reported

    p = purchase.copy()
    p["period"] = pd.to_datetime(p["Date"], dayfirst=True).dt.strftime("%m%Y")
    itc = (p.groupby("period")
             .agg(icgst=("CGST", "sum"), isgst=("SGST", "sum"), iigst=("IGST", "sum"))
             .round(2).reset_index())

    g3 = g1.merge(itc, on="period", how="left").fillna(0.0)
    g3["liability"] = (g3["cgst"] + g3["sgst"] + g3["igst"]).round(2)
    g3["itc_total"] = (g3["icgst"] + g3["isgst"] + g3["iigst"]).round(2)
    g3["paid_by_itc"] = np.minimum(g3["liability"], g3["itc_total"]).round(2)
    g3["paid_in_cash"] = (g3["liability"] - g3["paid_by_itc"]).round(2)
    g3.loc[5, "paid_in_cash"] = round(g3.loc[5, "paid_in_cash"] - 48_000, 2)   # short paid
    g3.loc[9, "paid_by_itc"] = round(g3.loc[9, "paid_by_itc"] * 0.5, 2)        # cash despite ITC
    g3.loc[9, "paid_in_cash"] = round(g3.loc[9, "liability"] - g3.loc[9, "paid_by_itc"], 2)

    gstr1 = pd.DataFrame({
        "Tax Period": g1["period"], "Taxable Value": g1["taxable"],
        "CGST": g1["cgst"], "SGST": g1["sgst"], "IGST": g1["igst"], "Cess": 0.0,
    })
    gstr3b = pd.DataFrame({
        "Tax Period": g3["period"], "Taxable Value": g3["taxable"],
        "CGST": g3["cgst"], "SGST": g3["sgst"], "IGST": g3["igst"], "Cess": 0.0,
        "ITC CGST": g3["icgst"], "ITC SGST": g3["isgst"], "ITC IGST": g3["iigst"],
        "Paid by ITC": g3["paid_by_itc"], "Paid in Cash": g3["paid_in_cash"],
        "Challan No": [f"CIN{int(x):06d}" for x in range(len(g3))],
        "Date of Payment": [(pd.Period(f"{p[2:]}-{p[:2]}", "M").to_timestamp()
                             + pd.Timedelta(days=49)).strftime("%d-%m-%Y")
                            for p in g3["period"]],
    })
    gstr3b.loc[3, "Challan No"] = ""                                  # missing challan
    return gstr1, gstr3b


# ------------------------------------------------ vouchers & outstanding

def build_vouchers(sales: pd.DataFrame, purchase: pd.DataFrame):
    """Receipts/payments against ~85% of invoices; the rest stay outstanding."""
    receipts, payments = [], []
    open_sales, open_purch = [], []

    for i, r in sales.iterrows():
        d = pd.to_datetime(r["Date"], dayfirst=True)
        cust = CUST[CUST["name"] == r["Party Name"]].iloc[0]
        if RNG.random() < 0.85 and d < END - pd.Timedelta(days=20):
            lag = int(RNG.integers(5, int(cust["credit_days"]) + 45))
            pay_date = min(d + pd.Timedelta(days=lag), END)
            cash = RNG.random() < 0.08
            receipts.append({
                "Date": pay_date.strftime("%d-%m-%Y"), "Party Name": r["Party Name"],
                "Amount": r["Total Invoice Value"],
                "Mode": "CASH" if cash else RNG.choice(["NEFT", "RTGS", "CHEQUE", "UPI"]),
                "Instrument No": "" if cash else f"UTR{int(RNG.integers(1e6, 9e6))}",
                "Against Bill": r["Invoice No"],
                "Receipt No": f"RCT-{i + 1:04d}",
                "Narration": f"Received from {r['Party Name']}",
            })
        else:
            open_sales.append((r, cust))

    for i, r in purchase.iterrows():
        d = pd.to_datetime(r["Date"], dayfirst=True)
        sup = SUPP[SUPP["name"] == r["Supplier Name"]].iloc[0]
        if RNG.random() < 0.82 and d < END - pd.Timedelta(days=20):
            lag = int(RNG.integers(5, int(sup["credit_days"]) + 40))
            pay_date = min(d + pd.Timedelta(days=lag), END)
            cash = RNG.random() < 0.07
            payments.append({
                "Date": pay_date.strftime("%d-%m-%Y"), "Party Name": r["Supplier Name"],
                "Amount": r["Invoice Value"],
                "Mode": "CASH" if cash else RNG.choice(["NEFT", "RTGS", "CHEQUE"]),
                "Instrument No": "" if cash else f"UTR{int(RNG.integers(1e6, 9e6))}",
                "Against Bill": r["Supplier Invoice No"],
                "Payment No": f"PMT-{i + 1:04d}",
                "Narration": f"Paid to {r['Supplier Name']}",
            })
        else:
            open_purch.append((r, sup))

    rec = pd.DataFrame(receipts)
    pay = pd.DataFrame(payments)

    # -- deliberate defects ---------------------------------------------
    dup = pay.iloc[20].copy()                       # duplicate payment 2 days later
    dup["Date"] = (pd.to_datetime(dup["Date"], dayfirst=True)
                   + pd.Timedelta(days=2)).strftime("%d-%m-%Y")
    dup["Payment No"] = "PMT-DUP1"
    pay = pd.concat([pay, pd.DataFrame([dup])], ignore_index=True)

    # cash payments above the 40A(3) limit, incl. a same-day split
    extra_pay = [
        {"Date": "12-07-2025", "Party Name": "Sathya Transport Service", "Amount": 48_500,
         "Mode": "CASH", "Instrument No": "", "Against Bill": "", "Payment No": "PMT-C001",
         "Narration": "Freight paid in cash"},
        {"Date": "19-09-2025", "Party Name": "Unity Engineering Works", "Amount": 9_800,
         "Mode": "CASH", "Instrument No": "", "Against Bill": "", "Payment No": "PMT-C002",
         "Narration": "Job work"},
        {"Date": "19-09-2025", "Party Name": "Unity Engineering Works", "Amount": 9_600,
         "Mode": "CASH", "Instrument No": "", "Against Bill": "", "Payment No": "PMT-C003",
         "Narration": "Job work"},
        {"Date": "05-01-2026", "Party Name": "Metro Office Supplies", "Amount": 2_50_000,
         "Mode": "NEFT", "Instrument No": "UTR7770001", "Against Bill": "",
         "Payment No": "PMT-R001", "Narration": ""},
    ]
    pay = pd.concat([pay, pd.DataFrame(extra_pay)], ignore_index=True)

    # a 269ST cash receipt
    rec = pd.concat([rec, pd.DataFrame([
        {"Date": "22-11-2025", "Party Name": "Coastal Trading Co", "Amount": 2_40_000,
         "Mode": "CASH", "Instrument No": "", "Against Bill": "", "Receipt No": "RCT-C001",
         "Narration": "Cash received on account"},
        {"Date": "03-02-2026", "Party Name": "Sunrise Traders", "Amount": 1_75_000,
         "Mode": "NEFT", "Instrument No": "UTR8880002", "Against Bill": "",
         "Receipt No": "RCT-A001", "Narration": "Advance"},
    ])], ignore_index=True)

    debtors = pd.DataFrame([{
        "Party Name": r["Party Name"], "Bill No": r["Invoice No"],
        "Bill Date": r["Date"], "Bill Amount": r["Total Invoice Value"],
        "Pending Amount": r["Total Invoice Value"], "Credit Days": int(c["credit_days"]),
    } for r, c in open_sales])
    creditors = pd.DataFrame([{
        "Party Name": r["Supplier Name"], "Bill No": r["Supplier Invoice No"],
        "Bill Date": r["Date"], "Bill Amount": r["Invoice Value"],
        "Pending Amount": r["Invoice Value"], "Credit Days": int(s["credit_days"]),
    } for r, s in open_purch])

    # advance received shows as a credit balance in debtors
    debtors = pd.concat([debtors, pd.DataFrame([{
        "Party Name": "Sunrise Traders", "Bill No": "ADV-001", "Bill Date": "03-02-2026",
        "Bill Amount": -1_75_000, "Pending Amount": -1_75_000, "Credit Days": 0}])],
        ignore_index=True)
    return rec, pay, debtors, creditors


# ------------------------------------------------------------ bank/cash

def build_bank(rec: pd.DataFrame, pay: pd.DataFrame, gstr3b: pd.DataFrame,
               salary: pd.DataFrame):
    lines = []
    for _, r in rec[rec["Mode"] != "CASH"].iterrows():
        lines.append({"Date": r["Date"], "Narration": f"NEFT CR-{r['Party Name'][:22]}",
                      "Chq/Ref No": r["Instrument No"], "Withdrawal Amt.": "",
                      "Deposit Amt.": r["Amount"]})
    for _, p in pay[pay["Mode"] != "CASH"].iterrows():
        lines.append({"Date": p["Date"], "Narration": f"NEFT DR-{p['Party Name'][:22]}",
                      "Chq/Ref No": p["Instrument No"], "Withdrawal Amt.": p["Amount"],
                      "Deposit Amt.": ""})
    for _, g in gstr3b.iterrows():
        if float(g["Paid in Cash"]) > 0:
            lines.append({"Date": g["Date of Payment"],
                          "Narration": f"GST CHALLAN PAYMENT {g['Tax Period']}",
                          "Chq/Ref No": g["Challan No"],
                          "Withdrawal Amt.": round(float(g["Paid in Cash"]), 2),
                          "Deposit Amt.": ""})
    sal_month = (salary[salary["Payment Mode"] != "CASH"]
                 .groupby("Month")["Net Pay"].sum().round(2))
    for month, amt in sal_month.items():
        d = (pd.to_datetime("01-" + month, format="%d-%b-%Y")
             + pd.offsets.MonthEnd(0) + pd.Timedelta(days=5))
        lines.append({"Date": d.strftime("%d-%m-%Y"), "Narration": "NEFT-SALARY PAYOUT",
                      "Chq/Ref No": f"SAL{month.replace('-', '')}",
                      "Withdrawal Amt.": amt, "Deposit Amt.": ""})

    # bank-originated entries that never reach the books
    for m in MONTHS:
        d = m + pd.offsets.MonthEnd(0)
        lines.append({"Date": d.strftime("%d-%m-%Y"), "Narration": "BANK CHARGES-AMC GST ON CHRG",
                      "Chq/Ref No": "", "Withdrawal Amt.": round(float(RNG.uniform(120, 850)), 2),
                      "Deposit Amt.": ""})
    for m in MONTHS[::3]:
        d = m + pd.offsets.MonthEnd(0)
        lines.append({"Date": d.strftime("%d-%m-%Y"), "Narration": "INTEREST CREDIT-FD",
                      "Chq/Ref No": "", "Withdrawal Amt.": "",
                      "Deposit Amt.": round(float(RNG.uniform(3000, 14000)), 2)})
    for m in MONTHS[::2]:
        d = m + pd.Timedelta(days=9)
        lines.append({"Date": d.strftime("%d-%m-%Y"), "Narration": "EMI TERM LOAN A/C 5567",
                      "Chq/Ref No": "", "Withdrawal Amt.": 1_25_000.0, "Deposit Amt.": ""})

    bank = pd.DataFrame(lines)
    bank["_d"] = pd.to_datetime(bank["Date"], dayfirst=True)
    bank = bank.sort_values("_d").reset_index(drop=True)

    # running balance
    bal, running = [], 25_00_000.0
    for _, r in bank.iterrows():
        running += float(r["Deposit Amt."] or 0) - float(r["Withdrawal Amt."] or 0)
        bal.append(round(running, 2))
    bank["Closing Balance"] = bal
    bank = bank.drop(columns=["_d"])

    # books side: the bank statement minus bank-originated items, plus three
    # cheques entered in the books that never got presented
    book = bank[~bank["Narration"].str.contains(
        "BANK CHARGES|INTEREST CREDIT|EMI TERM LOAN", regex=True)].copy()
    book = pd.DataFrame({
        "Date": book["Date"], "Particulars": book["Narration"],
        "Vch No": book["Chq/Ref No"],
        "Debit": book["Deposit Amt."], "Credit": book["Withdrawal Amt."],
    })
    opening_row = pd.DataFrame([
        {"Date": "01-04-2025", "Particulars": "Opening Balance b/f",
         "Vch No": "", "Debit": 25_00_000.0, "Credit": ""}])
    book = pd.concat([opening_row, book], ignore_index=True)
    unpresented = pd.DataFrame([
        {"Date": "18-12-2025", "Particulars": "Precision Tools India",
         "Vch No": "CHQ441201", "Debit": "", "Credit": 2_15_000.0},
        {"Date": "27-03-2026", "Particulars": "Orient Logistics LLP",
         "Vch No": "CHQ441288", "Debit": "", "Credit": 88_400.0},
        {"Date": "14-08-2025", "Particulars": "Skyline Advertising",
         "Vch No": "CHQ440915", "Debit": "", "Credit": 1_40_000.0},
    ])
    book = pd.concat([book, unpresented], ignore_index=True)
    return bank, book


def build_cashbook(rec: pd.DataFrame, pay: pd.DataFrame) -> pd.DataFrame:
    rows = [{"Date": "01-04-2025", "Particulars": "Opening cash balance",
             "Debit": 3_50_000.0, "Credit": ""}]
    for _, r in rec[rec["Mode"] == "CASH"].iterrows():
        rows.append({"Date": r["Date"], "Particulars": f"Cash received - {r['Party Name']}",
                     "Debit": r["Amount"], "Credit": ""})
    for _, p in pay[pay["Mode"] == "CASH"].iterrows():
        rows.append({"Date": p["Date"], "Particulars": f"Cash paid - {p['Party Name']}",
                     "Debit": "", "Credit": p["Amount"]})
    for m in MONTHS:
        rows.append({"Date": (m + pd.Timedelta(days=6)).strftime("%d-%m-%Y"),
                     "Particulars": "Petty expenses - sundries", "Debit": "",
                     "Credit": round(float(RNG.uniform(2500, 9500)), 2)})
    df = pd.DataFrame(rows)
    return df.sort_values(pd.to_datetime(df["Date"], dayfirst=True).name
                          if False else "Date", key=lambda s:
                          pd.to_datetime(s, dayfirst=True)).reset_index(drop=True)


# ------------------------------------------------------------- payroll

EMPLOYEES = [
    ("E001", "Ramesh Kumar", "Manager", 45000), ("E002", "Priya Nair", "Accountant", 32000),
    ("E003", "Suresh Babu", "Supervisor", 26000), ("E004", "Divya Rani", "Executive", 21000),
    ("E005", "Mohan Das", "Operator", 18000), ("E006", "Karthik R", "Operator", 17500),
    ("E007", "Lakshmi S", "Clerk", 15500), ("E008", "Arun Prakash", "Technician", 24000),
    ("E009", "Vinoth Kumar", "Driver", 14000), ("E010", "Sangeetha M", "HR Executive", 28000),
    ("E011", "Ganesh P", "Storekeeper", 16500), ("E012", "Fathima B", "Executive", 22000),
    ("E013", "Naveen Raj", "Sales Executive", 25000), ("E014", "Deepa V", "Clerk", 15000),
    ("E015", "Rajesh Iyer", "Engineer", 38000),
]


def build_salary() -> pd.DataFrame:
    rows = []
    for m in MONTHS:
        for emp, name, desig, base in EMPLOYEES:
            gross = float(base)
            if m.month == 10:                       # Diwali bonus month
                gross += base * 0.5
            if emp == "E009" and m.month in (7, 8):  # loss of pay
                gross = base * 0.6
            basic = round(gross * 0.5, 2)
            hra = round(gross * 0.2, 2)
            other = round(gross - basic - hra, 2)
            pf = round(min(basic, 15000) * 0.12, 0)
            esi = round(gross * 0.0075, 0) if gross <= 21000 else 0.0
            pt = 208.0 if gross > 21000 else 0.0
            tds = round(max(0.0, (gross - 50000) * 0.1), 0)
            ded = pf + esi + pt + tds
            rows.append({
                "Month": m.strftime("%b-%Y"), "Emp Code": emp, "Employee Name": name,
                "Designation": desig, "Date of Joining": "01-04-2022",
                "Paid Days": 30, "Basic": basic, "HRA": hra, "Other Allowance": other,
                "Gross Salary": round(gross, 2), "PF": pf, "ESI": esi,
                "Professional Tax": pt, "TDS": tds, "Advance": 0.0,
                "Other Deduction": 0.0, "Net Pay": round(gross - ded, 2),
                "Payment Mode": "NEFT",
                "Bank Account No": f"50100{int(emp[1:]):06d}",
                "PAN": pan(int(emp[1:]) * 37), "UAN": f"1012345{int(emp[1:]):05d}",
            })
    df = pd.DataFrame(rows)
    # -- deliberate defects --------------------------------------------
    df.loc[df["Emp Code"] == "E014", "Bank Account No"] = "50100000009"   # shared a/c
    idx = df[(df["Emp Code"] == "E005") & (df["Month"] == "Aug-2025")].index[0]
    df.loc[idx, "Net Pay"] = round(df.loc[idx, "Net Pay"] + 1750, 2)      # arithmetic
    idx = df[(df["Emp Code"] == "E011") & (df["Month"] == "Sep-2025")].index[0]
    df.loc[idx, "PF"] = 0.0                                               # PF not deducted
    idx = df[(df["Emp Code"] == "E007") & (df["Month"] == "Nov-2025")].index[0]
    df.loc[idx, "ESI"] = 0.0                                              # ESI not deducted
    mask = (df["Emp Code"] == "E009") & (df["Month"] == "Jan-2026")
    df.loc[mask, ["Payment Mode", "Net Pay"]] = ["CASH", 13_200.0]        # cash salary
    df.loc[df["Emp Code"] == "E013", "PAN"] = ""                          # no PAN
    return df


# ----------------------------------------------------------- inventory

ITEMS = [
    ("Cotton Yarn 30s", "KG", "Raw Material"), ("Polymer Granules PP", "KG", "Raw Material"),
    ("Corrugated Box 12x8", "NOS", "Packing"), ("Industrial Adhesive", "LTR", "Consumable"),
    ("Steel Rod 12mm", "KG", "Raw Material"), ("Fastener Set M8", "NOS", "Component"),
    ("Finished Fabric A", "MTR", "Finished Goods"), ("Finished Fabric B", "MTR", "Finished Goods"),
    ("Moulded Cap 28mm", "NOS", "Finished Goods"), ("Label Roll", "NOS", "Packing"),
    ("Lubricant Oil", "LTR", "Consumable"), ("Dye Chemical Blue", "KG", "Raw Material"),
    ("Dye Chemical Red", "KG", "Raw Material"), ("Machine Spare - Belt", "NOS", "Spares"),
    ("Machine Spare - Bearing", "NOS", "Spares"), ("Packing Tape", "NOS", "Packing"),
    ("Finished Sheet C", "NOS", "Finished Goods"), ("Scrap Cotton", "KG", "Scrap"),
    ("Scrap Polymer", "KG", "Scrap"), ("Solvent Grade B", "LTR", "Consumable"),
    ("Cartons Large", "NOS", "Packing"), ("Pallet Wooden", "NOS", "Packing"),
    ("Finished Cap Blue", "NOS", "Finished Goods"), ("Finished Cap Red", "NOS", "Finished Goods"),
    ("Obsolete Spare Kit", "NOS", "Spares"),
]


def build_inventory() -> pd.DataFrame:
    rows = []
    for i, (item, uom, grp) in enumerate(ITEMS):
        rate = float(np.round(RNG.uniform(40, 900), 2))
        oq = float(np.round(RNG.uniform(100, 3000), 0))
        iq = float(np.round(RNG.uniform(500, 9000), 0))
        oqt = float(np.round(RNG.uniform(400, min(iq + oq, 8500)), 0))
        cq = oq + iq - oqt
        rows.append({
            "Item Name": item, "Unit": uom, "Stock Group": grp,
            "Opening Qty": oq, "Opening Value": round(oq * rate, 2),
            "Inward Qty": iq, "Inward Value": round(iq * rate, 2),
            "Outward Qty": oqt, "Outward Value": round(oqt * rate * 1.18, 2),
            "Closing Qty": cq, "Closing Value": round(cq * rate, 2),
            "Rate": rate,
        })
    df = pd.DataFrame(rows)
    df.loc[4, ["Closing Qty", "Closing Value"]] = [-120.0, -32_400.0]      # negative stock
    df.loc[11, "Closing Qty"] = df.loc[11, "Closing Qty"] + 260            # movement variance
    df.loc[24, ["Outward Qty", "Outward Value"]] = [0.0, 0.0]              # non-moving
    df.loc[24, "Closing Qty"] = df.loc[24, "Opening Qty"] + df.loc[24, "Inward Qty"]
    df.loc[8, "Closing Value"] = round(df.loc[8, "Closing Value"] * 1.35, 2)  # rate variance
    return df


def build_conversion() -> pd.DataFrame:
    pairs = [("Finished Fabric A", "Cotton Yarn 30s", 92.0),
             ("Finished Fabric B", "Cotton Yarn 30s", 90.0),
             ("Moulded Cap 28mm", "Polymer Granules PP", 95.0),
             ("Finished Cap Blue", "Polymer Granules PP", 94.0),
             ("Finished Sheet C", "Steel Rod 12mm", 88.0)]
    rows = []
    for k in range(40):
        out_item, in_item, std = pairs[k % len(pairs)]
        d = rnd_date(MONTHS[k % 12])
        in_qty = float(np.round(RNG.uniform(500, 4000), 0))
        drift = RNG.normal(0, 3.2)
        if k in (7, 19):
            drift = -14.0                                   # abnormal loss
        if k == 28:
            drift = 11.0                                    # unrecorded input
        actual = max(40.0, std + drift)
        out_qty = round(in_qty * actual / 100, 0)
        in_rate = float(np.round(RNG.uniform(60, 240), 2))
        rows.append({
            "Date": d.strftime("%d-%m-%Y"), "Voucher No": f"MFG-{k + 1:03d}",
            "Finished Item": out_item, "Output Qty": out_qty,
            "Output Value": round(out_qty * in_rate * 1.45, 2),
            "Raw Material": in_item, "Input Qty": in_qty,
            "Input Value": round(in_qty * in_rate, 2), "Standard Yield": std,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- writing

def write_bank_csv(bank: pd.DataFrame, path: Path) -> None:
    """Write with the banner rows a real HDFC export carries, to exercise the
    header auto-detection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("HDFC BANK LIMITED\n")
        fh.write("Statement of account - 50200012345678\n")
        fh.write("Period: 01/04/2025 to 31/03/2026\n")
        fh.write("\n")
        bank.to_csv(fh, index=False, lineterminator="\n")


def write_portal_xlsx(df: pd.DataFrame, path: Path, banner: list[str]) -> None:
    """GST portal downloads put several banner rows above the real header."""
    pad = pd.DataFrame([[b] + [""] * (len(df.columns) - 1) for b in banner],
                       columns=df.columns)
    header = pd.DataFrame([list(df.columns)], columns=df.columns)
    body = pd.concat([pad, header, df.astype(object)], ignore_index=True)
    body.to_excel(path, index=False, header=False)


def main() -> None:
    IN.mkdir(parents=True, exist_ok=True)
    (IN / "bank").mkdir(exist_ok=True)

    sales = build_sales()
    purchase = build_purchase()
    gstr2b = build_gstr2b(purchase)
    gstr1, gstr3b = build_gst_returns(sales, purchase)
    rec, pay, debtors, creditors = build_vouchers(sales, purchase)
    salary = build_salary()
    bank, book = build_bank(rec, pay, gstr3b, salary)
    cashbook = build_cashbook(rec, pay)
    inventory = build_inventory()
    conversion = build_conversion()

    write_bank_csv(bank, IN / "bank" / "hdfc_statement_2025-26.csv")
    write_portal_xlsx(gstr2b, IN / "gstr2b.xlsx",
                      ["GSTR-2B Auto-drafted ITC statement",
                       "GSTIN: 33AABCS1429B1Z1   Period: Apr 2025 - Mar 2026",
                       "B2B - Invoices"])
    for df, name in [(sales, "sales_register"), (purchase, "purchase_register"),
                     (gstr1, "gstr1"), (gstr3b, "gstr3b"), (rec, "receipts"),
                     (pay, "payments"), (debtors, "debtors_outstanding"),
                     (creditors, "creditors_outstanding"), (salary, "salary_register"),
                     (inventory, "inventory"), (conversion, "conversion"),
                     (cashbook, "cash_book"), (book, "bank_book")]:
        df.to_excel(IN / f"{name}.xlsx", index=False)

    print(f"Sample data written to {IN}")
    for f in sorted(IN.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(ROOT)}  ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
