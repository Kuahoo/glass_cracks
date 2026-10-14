"""Filesystem locations. Everything is resolved from DATA_ROOT and OUT_ROOT so the same
code runs on the laptop, Colab and the cluster with only environment variables changed."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", REPO_ROOT / "temp" / "glass_crack_dataset_bundle")).resolve()


def out_root() -> Path:
    return Path(os.environ.get("OUT_ROOT", REPO_ROOT / "outputs")).resolve()


def merged_dir() -> Path:
    return data_root() / "processed" / "glass_merged_seg"


def linear_dir() -> Path:
    return data_root() / "processed" / "glass_merged_linear"


def ncugd_dir() -> Path:
    return data_root() / "processed" / "ncugd_external_test"


def agdd_dir() -> Path:
    return data_root() / "datasets" / "agdd_aircraft_glass" / "data"


def cubit_dir() -> Path:
    return data_root() / "datasets" / "cubit_inseg"


def results_csv() -> Path:
    return out_root() / "results" / "results.csv"


def resolve(path: str | Path) -> Path:
    """Expand ${DATA_ROOT}/${OUT_ROOT} placeholders used in config files."""
    s = str(path).replace("${DATA_ROOT}", str(data_root())).replace("${OUT_ROOT}", str(out_root()))
    return Path(os.path.expandvars(s)).expanduser()
