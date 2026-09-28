"""Inventory movement and conversion (manufacturing) checks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..normalize import r2


def _movement(inv: pd.DataFrame) -> pd.DataFrame:
    d = inv.copy()
    d["derived_closing_qty"] = (d["opening_qty"] + d["inward_qty"] - d["outward_qty"]).round(3)
    d["qty_variance"] = (d["closing_qty"] - d["derived_closing_qty"]).round(3)
    d["derived_closing_val"] = (d["opening_val"] + d["inward_val"] - d["outward_val"]).round(2)
    d["val_variance"] = (d["closing_val"] - d["derived_closing_val"]).round(2)
    d["closing_rate"] = np.where(d["closing_qty"] != 0,
                                 (d["closing_val"] / d["closing_qty"]).round(2), 0.0)
    d["inward_rate"] = np.where(d["inward_qty"] != 0,
                                (d["inward_val"] / d["inward_qty"]).round(2), 0.0)
    d["rate_variance %"] = np.where(
        d["inward_rate"] > 0,
        (100 * (d["closing_rate"] - d["inward_rate"]) / d["inward_rate"]).round(2), 0.0)
    consumed = (d["opening_qty"] + d["inward_qty"]).replace(0, np.nan)
    d["turnover_ratio"] = (d["outward_qty"] / consumed).round(2)
    d["negative_stock"] = d["closing_qty"] < 0
    d["negative_value"] = d["closing_val"] < 0
    return d


def _conversion(conv: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Yield per production voucher vs the standard/BOM yield."""
    d = conv.copy()
    d["actual_yield %"] = np.where(
        d["input_qty"] != 0, (100 * d["output_qty"] / d["input_qty"]).round(2), 0.0)
    std = d["std_yield"].replace(0, np.nan)
    d["std_yield %"] = std.fillna(d["actual_yield %"].median() if len(d) else 0).round(2)
    d["yield_variance %"] = (d["actual_yield %"] - d["std_yield %"]).round(2)
    d["conversion_cost"] = (d["output_val"] - d["input_val"]).round(2)
    d["cost_per_unit"] = np.where(d["output_qty"] != 0,
                                  (d["output_val"] / d["output_qty"]).round(2), 0.0)
    tol = float(cfg.th("yield_variance_pct", 5.0))
    d["yield_flag"] = np.where(d["yield_variance %"] < -tol, "LOW YIELD",
                               np.where(d["yield_variance %"] > tol, "HIGH YIELD", "NORMAL"))
    return d


def run(data: dict[str, pd.DataFrame], cfg: Config) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    inv = data.get("inventory", pd.DataFrame())
    conv = data.get("conversion", pd.DataFrame())

    if inv is not None and not inv.empty:
        d = _movement(inv)
        out["inventory_detail"] = d.sort_values("closing_val", ascending=False)
        out["inventory_summary"] = pd.DataFrame([{
            "Items": len(d),
            "Opening value": r2(d["opening_val"].sum()),
            "Inward value": r2(d["inward_val"].sum()),
            "Outward value": r2(d["outward_val"].sum()),
            "Closing value": r2(d["closing_val"].sum()),
            "Derived closing value": r2(d["derived_closing_val"].sum()),
            "Value variance": r2(d["val_variance"].sum()),
            "Items with qty variance": int((d["qty_variance"].abs() > 0.001).sum()),
            "Items with negative stock": int(d["negative_stock"].sum()),
            "Items with negative value": int(d["negative_value"].sum()),
            "Slow moving (turnover < 0.2)": int((d["turnover_ratio"] < 0.2).sum()),
        }])
        out["inventory_group"] = (d.groupby("group", dropna=False)
                                   [["opening_val", "inward_val", "outward_val", "closing_val"]]
                                   .sum().round(2).reset_index()
                                   .sort_values("closing_val", ascending=False))

    if conv is not None and not conv.empty:
        c = _conversion(conv, cfg)
        out["conversion_detail"] = c.sort_values("yield_variance %")
        out["conversion_summary"] = pd.DataFrame([{
            "Production vouchers": len(c),
            "Input value": r2(c["input_val"].sum()),
            "Output value": r2(c["output_val"].sum()),
            "Conversion cost added": r2(c["conversion_cost"].sum()),
            "Average yield %": r2(c["actual_yield %"].mean()),
            "Low yield vouchers": int((c["yield_flag"] == "LOW YIELD").sum()),
            "High yield vouchers": int((c["yield_flag"] == "HIGH YIELD").sum()),
        }])
        out["conversion_item"] = (c.groupby("output_item")
                                   .agg(vouchers=("output_qty", "size"),
                                        input_qty=("input_qty", "sum"),
                                        output_qty=("output_qty", "sum"),
                                        avg_yield=("actual_yield %", "mean"),
                                        conversion_cost=("conversion_cost", "sum"))
                                   .round(2).reset_index())
    return out
