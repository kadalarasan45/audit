"""Configuration loading + a thin accessor with dotted lookups."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"
DEFAULT_COLMAP = ROOT / "config" / "column_maps.yaml"


class Config:
    """Holds config.yaml + column_maps.yaml and answers dotted questions."""

    def __init__(self, data: dict, colmap: dict, path: Path | None = None):
        self._d = data
        self.colmap = colmap
        self.path = path

    # -- construction -------------------------------------------------
    @classmethod
    def load(cls, config_path: str | Path | None = None,
             colmap_path: str | Path | None = None) -> "Config":
        cp = Path(config_path or DEFAULT_CONFIG)
        mp = Path(colmap_path or DEFAULT_COLMAP)
        with open(cp) as fh:
            data = yaml.safe_load(fh) or {}
        with open(mp) as fh:
            colmap = yaml.safe_load(fh) or {}
        return cls(data, colmap, cp)

    # -- lookups ------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    def th(self, name: str, default: Any = None) -> Any:
        """Shorthand for ``thresholds.<name>``."""
        return self.get(f"thresholds.{name}", default)

    def keywords(self, name: str) -> list[str]:
        return [str(k).lower() for k in self.get(f"classification.{name}", []) or []]

    def aliases(self, section: str) -> dict[str, list[str]]:
        return self.colmap.get(section, {}) or {}

    # -- derived ------------------------------------------------------
    @property
    def client(self) -> str:
        return str(self.get("client.name", "Client"))

    @property
    def fy(self) -> str:
        return str(self.get("client.fy", ""))

    @property
    def period_from(self) -> pd.Timestamp:
        return pd.Timestamp(self.get("client.period_from", "1900-01-01"))

    @property
    def period_to(self) -> pd.Timestamp:
        return pd.Timestamp(self.get("client.period_to", "2099-12-31"))

    @property
    def state_code(self) -> str:
        return str(self.get("client.state_code", "")).zfill(2)

    @property
    def holidays(self) -> set[pd.Timestamp]:
        return {pd.Timestamp(h).normalize() for h in (self.get("holidays") or [])}

    @property
    def outdir(self) -> Path:
        p = ROOT / str(self.get("report.output_dir", "output"))
        p.mkdir(parents=True, exist_ok=True)
        return p

    def source_patterns(self, name: str) -> list[str]:
        pats = self.get(f"sources.{name}", []) or []
        return [pats] if isinstance(pats, str) else list(pats)

    def in_period(self, ts) -> bool:
        if ts is None or pd.isna(ts):
            return False
        return self.period_from <= pd.Timestamp(ts) <= self.period_to
