"""Experiment config loading (YAML with ${DATA_ROOT} / ${OUT_ROOT} expansion and an optional `base:`)."""
from __future__ import annotations

from pathlib import Path

import yaml

from glass_crack_models import paths

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists() and (CONFIG_DIR / p).exists():
        p = CONFIG_DIR / p
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if "base" in cfg:
        base = load_config(p.parent / cfg.pop("base"))
        base.update(cfg)
        cfg = base
    cfg["_config_path"] = str(p.resolve())
    return cfg


def resolved(value):
    return paths.resolve(value) if isinstance(value, str) and "${" in value else value
