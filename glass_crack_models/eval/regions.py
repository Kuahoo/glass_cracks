"""Regions (instances) as a bounding box plus a cropped boolean mask, and score-map -> regions."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from glass_crack_models.protocol import MIN_REGION_PX


@dataclass
class Region:
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive
    mask: np.ndarray  # bool, shape (y1 - y0, x1 - x0)
    score: float = 1.0
    label: int = 0

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def full(self, h: int, w: int) -> np.ndarray:
        m = np.zeros((h, w), bool)
        m[self.y0:self.y1, self.x0:self.x1] = self.mask
        return m

    def window(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """This region's mask rendered into the window [x0, x1) x [y0, y1)."""
        out = np.zeros((y1 - y0, x1 - x0), bool)
        ix0, iy0, ix1, iy1 = max(x0, self.x0), max(y0, self.y0), min(x1, self.x1), min(y1, self.y1)
        if ix0 < ix1 and iy0 < iy1:
            out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = self.mask[iy0 - self.y0:iy1 - self.y0, ix0 - self.x0:ix1 - self.x0]
        return out


def region_from_mask(mask: np.ndarray, score: float = 1.0, label: int = 0) -> Region | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    return Region(x0, y0, x1, y1, mask[y0:y1, x0:x1].astype(bool), score, label)


def extract_regions(score_map: np.ndarray, thr: float, min_area: int = MIN_REGION_PX) -> list[Region]:
    """8-connected components of score_map >= thr, dropping those smaller than min_area.
    Region score = max score inside the component."""
    binary = (score_map >= thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        crop = lab[y:y + h, x:x + w] == i
        out.append(Region(int(x), int(y), int(x + w), int(y + h), crop, float(score_map[y:y + h, x:x + w][crop].max())))
    return out


def suppress_small(score_map: np.ndarray, thr: float, min_area: int = MIN_REGION_PX) -> np.ndarray:
    """Zero the score of components (at thr) smaller than min_area; used per scale before stitching."""
    binary = (score_map >= thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] < min_area
    out = score_map.copy()
    out[small[lab]] = 0.0
    return out


def union_mask(regions: list[Region], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    for r in regions:
        m[r.y0:r.y1, r.x0:r.x1] |= r.mask
    return m
