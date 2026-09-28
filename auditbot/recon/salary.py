"""Salary expense processing.

Recomputes the pay register arithmetic, tests PF/ESI/PT against statutory
ceilings, tracks month-on-month movement and ties the payroll cost back to
what actually left the bank.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..matching import Pass, cascade_match
from ..normalize import r2, validate_pan


def _statutory(sal: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Expected PF/ESI per the wage ceilings, next to what was deducted."""
    pf_ceiling = float(cfg.th("pf_wage_ceiling", 15000))
    esi_ceiling = float(cfg.th("esi_wage_ceiling", 21000))
    pf_pct = float(cfg.th("pf_employee_pct", 12.0)) / 100
    esi_pct = float(cfg.th("esi_employee_pct", 0.75)) / 100

    d = sal.copy()
    pf_base = d["basic"].where(d["basic"] > 0, d["gross"])
    d["pf_expected"] = (np.minimum(pf_base, pf_ceiling) * pf_pct).round(0)
    d.loc[pf_base <= 0, "pf_expected"] = 0.0
    d["pf_variance"] = (d["pf_employee"] - d["pf_expected"]).round(2)

    esi_applicable = d["gross"] <= esi_ceiling
    d["esi_expected"] = np.where(esi_applicable, (d["gross"] * esi_pct).round(0), 0.0)
    d["esi_variance"] = (d["esi_employee"] - d["esi_expected"]).round(2)
    d["esi_applicable"] = esi_applicable
    return d


def _monthly(sal: pd.DataFrame) -> pd.DataFrame:
    g = (sal.groupby("month")
            .agg(headcount=("emp_key", "nunique"),
                 gross=("gross", "sum"),
                 pf=("pf_employee", "sum"),
                 esi=("esi_employee", "sum"),
                 pt=("pt", "sum"),
                 tds=("tds", "sum"),
                 deductions=("total_deduct", "sum"),
                 net=("net", "sum"),
                 cash_paid=("net", lambda s: r2(sal.loc[s.index][sal.loc[s.index, "is_cash"]]["net"].sum())))
            .round(2).reset_index().sort_values("month"))
    g["gross_change"] = g["gross"].diff().round(2)
    g["gross_change %"] = (100 * g["gross"].pct_change()).round(2)
    g["headcount_change"] = g["headcount"].diff()
    g["avg_cost_per_head"] = (g["gross"] / g["headcount"].replace(0, np.nan)).round(2)
    return g


def _employee(sal: pd.DataFrame) -> pd.DataFrame:
    g = (sal.groupby(["emp_key", "emp_name"], dropna=False)
            .agg(months_paid=("month", "nunique"),
                 gross=("gross", "sum"),
                 net=("net", "sum"),
                 tds=("tds", "sum"),
                 pf=("pf_employee", "sum"),
                 avg_gross=("gross", "mean"),
                 min_gross=("gross", "min"),
                 max_gross=("gross", "max"),
                 pan=("pan", "first"),
                 uan=("uan", "first"),
                 bank_acct=("bank_acct", "first"))
            .round(2).reset_index().sort_values("gross", ascending=False))
    g["gross_swing %"] = np.where(
        g["min_gross"] > 0,
        (100 * (g["max_gross"] - g["min_gross"]) / g["min_gross"]).round(2), 0.0)
    g["pan_valid"] = g["pan"].map(validate_pan)
    return g


def bank_tie_up(sal: pd.DataFrame, bank: pd.DataFrame,
                cfg: Config) -> dict[str, pd.DataFrame]:
    """Monthly payroll net pay vs salary debits in the bank statement."""
    if bank is None or bank.empty:
        return {}
    sal_bank = bank[bank["category"].eq("SALARY") & (bank["amount"] < 0)].copy()
    if sal_bank.empty:
        return {}
    sal_bank["month"] = sal_bank["txn_date"].dt.to_period("M").astype(str)
    b = sal_bank.groupby("month")["amount"].sum().abs().round(2).rename("bank_salary_debit")
    p = sal[~sal["is_cash"]].groupby("month")["net"].sum().round(2).rename("payroll_net")
    m = pd.concat([p, b], axis=1).fillna(0.0).reset_index()
    m["difference"] = (m["payroll_net"] - m["bank_salary_debit"]).round(2)
    m["status"] = np.where(m["difference"].abs() <= 1.0, "AGREED", "DIFFERENCE")
    return {"salary_bank_tieup": m}


def run(data: dict[str, pd.DataFrame], cfg: Config) -> dict[str, pd.DataFrame]:
    sal = data.get("salary", pd.DataFrame())
    if sal is None or sal.empty:
        return {}
    d = _statutory(sal, cfg)
    d["net_variance"] = (d["net"] - d["computed_net"]).round(2)

    summary = pd.DataFrame([{
        "Months processed": d["month"].nunique(),
        "Employees": d["emp_key"].nunique(),
        "Pay lines": len(d),
        "Gross salary": r2(d["gross"].sum()),
        "Total deductions": r2(d["total_deduct"].sum()),
        "Net paid": r2(d["net"].sum()),
        "PF deducted": r2(d["pf_employee"].sum()),
        "ESI deducted": r2(d["esi_employee"].sum()),
        "TDS deducted": r2(d["tds"].sum()),
        "Paid in cash": r2(d.loc[d["is_cash"], "net"].sum()),
        "Net pay arithmetic errors": int((d["net_variance"].abs() > 1).sum()),
        "PF variances": int((d["pf_variance"].abs() > 1).sum()),
        "ESI variances": int((d["esi_variance"].abs() > 1).sum()),
    }])

    out = {"salary_detail": d, "salary_summary": summary,
           "salary_monthly": _monthly(d), "salary_employee": _employee(d)}
    out.update(bank_tie_up(d, data.get("bank"), cfg))
    return out
