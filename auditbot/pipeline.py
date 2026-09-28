"""Orchestrator: ingest -> reconcile -> flag exceptions -> report."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import Config
from .exceptions import build_register, register_summary, rule_catalogue
from .ingest import load_all
from .normalize import r2
from .recon import ageing, bank, billwise, gst, inventory, salary, trade

log = logging.getLogger("auditbot.pipeline")

STAGES = (
    ("Bank reconciliation", bank.run),
    ("Debtors / creditors ageing", ageing.run),
    ("Bill-wise receipts & payments", billwise.run),
    ("GST reconciliation", gst.run),
    ("Purchase & sales reconciliation", trade.run),
    ("Salary processing", salary.run),
    ("Inventory & conversion", inventory.run),
)


@dataclass
class Result:
    cfg: Config
    data: dict[str, pd.DataFrame] = field(default_factory=dict)
    recon: dict[str, pd.DataFrame] = field(default_factory=dict)
    register: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    kpis: dict = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)
    outdir: Path = Path(".")

    def get(self, key: str) -> pd.DataFrame:
        return self.recon.get(key, pd.DataFrame())


def _kpis(res: Result) -> dict:
    cfg, data, recon, reg = res.cfg, res.data, res.recon, res.register

    def s(key: str, col: str) -> float:
        df = data.get(key)
        return r2(df[col].sum()) if df is not None and not df.empty and col in df else 0.0

    def first(key: str, col: str, default=0):
        df = recon.get(key)
        if df is None or df.empty or col not in df.columns:
            return default
        return df.iloc[0][col]

    high = int((reg["severity"] == "HIGH").sum()) if not reg.empty else 0
    med = int((reg["severity"] == "MEDIUM").sum()) if not reg.empty else 0
    low = int((reg["severity"] == "LOW").sum()) if not reg.empty else 0

    return {
        "client": cfg.client,
        "fy": cfg.fy,
        "period": f"{cfg.period_from.date()} to {cfg.period_to.date()}",
        "generated": pd.Timestamp.now().strftime("%d-%b-%Y %H:%M"),
        "sales": s("sales", "total"),
        "purchases": s("purchase", "total"),
        "receipts": s("receipts", "amount"),
        "payments": s("payments", "amount"),
        "debtors": s("debtors", "balance"),
        "creditors": s("creditors", "balance"),
        "salary_cost": s("salary", "gross"),
        "closing_stock": s("inventory", "closing_val"),
        "bank_lines": len(data.get("bank", pd.DataFrame())),
        "bank_match_rate": first("bank_summary", "Match rate %", 0.0),
        "bank_unreconciled": first("bank_summary", "Unreconciled difference", 0.0),
        "bank_only": len(recon.get("bank_only", pd.DataFrame())),
        "book_only": len(recon.get("book_only", pd.DataFrame())),
        "itc_at_risk": first("itc_stats", "ITC at risk", 0.0),
        "itc_unclaimed": first("itc_stats", "ITC unclaimed", 0.0),
        "gst_match_rate": first("itc_stats", "Match rate %", 0.0),
        "gst_shortfall": first("gst_payment_totals", "Net shortfall", 0.0),
        "debtors_overdue": first("debtors_summary", "Overdue amount", 0.0),
        "creditors_overdue": first("creditors_summary", "Overdue amount", 0.0),
        "unapplied_receipts": r2(recon.get("receipts_unapplied", pd.DataFrame())
                                 .get("unapplied_amount", pd.Series(dtype=float)).sum()),
        "unapplied_payments": r2(recon.get("payments_unapplied", pd.DataFrame())
                                 .get("unapplied_amount", pd.Series(dtype=float)).sum()),
        "cash_above_limit": int(reg[reg["code"].str.startswith("CSH")].shape[0])
                            if not reg.empty else 0,
        "duplicates": int(reg[reg["code"].str.startswith("DUP")].shape[0])
                      if not reg.empty else 0,
        "exceptions_total": len(reg),
        "exceptions_high": high,
        "exceptions_medium": med,
        "exceptions_low": low,
        "exception_value": r2(reg["amount"].abs().sum()) if not reg.empty else 0.0,
        "rules_run": len(rule_catalogue()),
    }


def run(cfg: Config | None = None, config_path: str | None = None) -> Result:
    cfg = cfg or Config.load(config_path)
    res = Result(cfg=cfg, outdir=cfg.outdir)

    t0 = time.time()
    res.data = load_all(cfg)
    res.log.append({"stage": "Ingestion", "seconds": round(time.time() - t0, 2),
                    "detail": ", ".join(f"{k}={len(v)}" for k, v in res.data.items()
                                        if v is not None and not v.empty) or "no data"})
    log.info("ingested: %s", res.log[-1]["detail"])

    for label, fn in STAGES:
        t = time.time()
        try:
            out = fn(res.data, cfg) or {}
            res.recon.update(out)
            detail = ", ".join(f"{k}={len(v)}" for k, v in out.items()
                               if v is not None and not v.empty) or "nothing produced"
        except Exception as exc:
            detail = f"FAILED: {exc}"
            log.error("%s failed: %s", label, exc, exc_info=True)
        res.log.append({"stage": label, "seconds": round(time.time() - t, 2),
                        "detail": detail})
        log.info("%-32s %.2fs  %s", label, time.time() - t, detail)

    t = time.time()
    res.register = build_register(res.data, res.recon, cfg)
    res.summary = register_summary(res.register)
    res.log.append({"stage": "Exception rules", "seconds": round(time.time() - t, 2),
                    "detail": f"{len(res.register)} findings from "
                              f"{len(rule_catalogue())} rules"})
    res.kpis = _kpis(res)
    log.info("total %.2fs, %d exceptions", time.time() - t0, len(res.register))
    return res
