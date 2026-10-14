"""Sliding-window inference with overlap blending, and aspect-preserving multi-scale passes."""
from __future__ import annotations

import cv2
import numpy as np

from glass_crack_models.eval.regions import suppress_small
from glass_crack_models.protocol import MIN_REGION_PX, TILE, TILE_OVERLAP


def tile_starts(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, int(round(tile * (1 - overlap))))
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def blend_window(h: int, w: int, floor: float = 0.05) -> np.ndarray:
    """Weights that fall off towards tile edges, so overlapping tiles hand over smoothly."""
    def ramp(n):
        x = np.minimum(np.arange(n) + 0.5, n - np.arange(n) - 0.5) / (n / 2)
        return np.clip(x, floor, 1.0)
    return np.outer(ramp(h), ramp(w)).astype(np.float32)


def _pad_to(img: np.ndarray, th: int, tw: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h >= th and w >= tw:
        return img
    mode = cv2.BORDER_REFLECT_101 if min(h, w) > 1 else cv2.BORDER_CONSTANT
    return cv2.copyMakeBorder(img, 0, max(0, th - h), 0, max(0, tw - w), mode)


def predict_tiled(predictor, img: np.ndarray, tile: int = TILE, overlap: float = TILE_OVERLAP,
                  batch: int = 8) -> dict[str, np.ndarray]:
    """Score maps at the image's own resolution. Images no larger than one tile run in a single pass
    (reflect-padded to the tile size if smaller), so 640x640 eval images see exactly what training saw."""
    h, w = img.shape[:2]
    padded = _pad_to(img, max(h, tile), max(w, tile))
    ph, pw = padded.shape[:2]
    ys, xs = tile_starts(ph, tile, overlap), tile_starts(pw, tile, overlap)
    if len(ys) == 1 and len(xs) == 1:
        maps = predictor.predict_batch([padded[:tile, :tile]])[0]
        return {k: v[:h, :w].astype(np.float32) for k, v in maps.items()}
    acc = {c: np.zeros((ph, pw), np.float32) for c in predictor.eval_classes}
    wsum = np.zeros((ph, pw), np.float32)
    win = blend_window(tile, tile)
    coords = [(y, x) for y in ys for x in xs]
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        preds = predictor.predict_batch([padded[y:y + tile, x:x + tile] for y, x in chunk])
        for (y, x), maps in zip(chunk, preds):
            for c, m in maps.items():
                acc[c][y:y + tile, x:x + tile] += m * win
            wsum[y:y + tile, x:x + tile] += win
    return {c: (a / np.maximum(wsum, 1e-6))[:h, :w] for c, a in acc.items()}


def auto_scales(h: int, w: int, tile: int = TILE) -> list[float]:
    """1.0 plus halvings while the long side still covers at least one tile."""
    scales, s = [1.0], 0.5
    while max(h, w) * s >= tile:
        scales.append(s)
        s /= 2
    return scales


def predict_multiscale(predictor, img: np.ndarray, scales: list[float], thr: float | None = None,
                       min_area: int = MIN_REGION_PX, tile: int = TILE, overlap: float = TILE_OVERLAP,
                       batch: int = 8) -> dict[str, np.ndarray]:
    """Run each scale (aspect preserved), drop regions under `min_area` at that scale's model-input
    resolution when `thr` is given, upsample to native size and combine by per-pixel max."""
    h, w = img.shape[:2]
    out: dict[str, np.ndarray] = {}
    for s in scales:
        if s == 1.0:
            im = img
        else:
            im = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                            interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        maps = predict_tiled(predictor, im, tile, overlap, batch)
        for c, m in maps.items():
            if thr is not None:
                m = suppress_small(m, thr, min_area)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            out[c] = m if c not in out else np.maximum(out[c], m)
    return out
