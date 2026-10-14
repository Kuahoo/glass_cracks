"""Cluster bootstrap over near-duplicate groups (plan Section 4).

Per-image additive counts are summed per group; groups are resampled with replacement and the
statistic is recomputed from the summed counts of each resample.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from glass_crack_models.protocol import BOOTSTRAP_CI, BOOTSTRAP_RESAMPLES


def group_sums(records: list[dict], groups: list, fields: list[str]) -> np.ndarray:
    keys = sorted(set(groups), key=str)
    index = {g: i for i, g in enumerate(keys)}
    out = np.zeros((len(keys), len(fields)))
    for rec, g in zip(records, groups):
        out[index[g]] += [rec.get(f, 0) for f in fields]
    return out


def cluster_bootstrap(records: list[dict], groups: list, stat: Callable[[dict], float],
                      n: int = BOOTSTRAP_RESAMPLES, ci: float = BOOTSTRAP_CI, seed: int = 0) -> tuple[float, float, float]:
    """Return (point estimate, lower, upper). `stat` maps {field: summed value} -> float (nan allowed)."""
    fields = sorted({k for r in records for k, v in r.items() if isinstance(v, (int, float, np.integer, np.floating))})
    sums = group_sums(records, groups, fields)
    point = stat(dict(zip(fields, sums.sum(0))))
    g = len(sums)
    if g == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(g, np.full(g, 1.0 / g), size=n)
    resampled = weights @ sums
    vals = np.array([stat(dict(zip(fields, row))) for row in resampled], dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float(point), float("nan"), float("nan")
    a = (1 - ci) / 2
    return float(point), float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))
