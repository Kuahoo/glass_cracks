"""Experiment config loading (YAML with an optional `base:`; nested dicts merge one level deep)."""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists() and (CONFIG_DIR / p).exists():
        p = CONFIG_DIR / p
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if "base" in cfg:
        base = load_config(p.parent / cfg.pop("base"))
        for k, v in cfg.items():  # nested dicts (e.g. `train:`) are merged, so a child lists only what it changes
            base[k] = {**base[k], **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        cfg = base
    cfg["_config_path"] = str(p.resolve())
    return cfg
