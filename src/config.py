"""Central configuration loading, path resolution and risk-category helpers."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "config.yaml"

RESEARCH_DISCLAIMER = (
    "Research prototype. All default records are SYNTHETIC (simulated) and were not collected "
    "from any football club. Predicted probabilities and risk categories are research outputs, "
    "not medical diagnoses, clinical risk scores or return-to-play decisions."
)


def _deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg = copy.deepcopy(cfg)
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] (relative paths are relative to the project root)."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def risk_category(prob: float, cfg: dict) -> str:
    """Map a probability in [0, 1] to a configurable illustrative risk category."""
    if prob is None or prob != prob or not (0.0 <= float(prob) <= 1.0):
        raise ValueError(f"Probability must be within [0, 1], got {prob!r}")
    cats = sorted(cfg["risk_categories"], key=lambda c: c["lower"])
    name = cats[0]["name"]
    for c in cats:
        if prob >= c["lower"]:
            name = c["name"]
    return name


RISK_COLOURS = {"Low": "#2EE59D", "Moderate": "#F5D547", "Elevated": "#F5A247", "High": "#F0616D"}
