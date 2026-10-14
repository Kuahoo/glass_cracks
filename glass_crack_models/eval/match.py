"""Fixed instance-matching rule (plan Section 4).

Lenient (headline): a prediction matches a GT instance if mask IoU >= 0.3, or if the two masks
overlap after both are dilated by 3 px. Strict: mask IoU >= 0.5 only.
Matching is greedy and one-to-one, predictions taken in descending score order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from glass_crack_models.eval.regions import Region
from glass_crack_models.protocol import MATCH_DILATE_PX, MATCH_IOU, STRICT_IOU


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    pairs: list[tuple[int, int, float]] = field(default_factory=list)  # (pred idx, gt idx, iou)
    unmatched_pred: list[int] = field(default_factory=list)
    unmatched_gt: list[int] = field(default_factory=list)


def _boxes_near(a: Region, b: Region, pad: int) -> bool:
    return a.x0 - pad < b.x1 + pad and b.x0 - pad < a.x1 + pad and a.y0 - pad < b.y1 + pad and b.y0 - pad < a.y1 + pad


def pair_stats(a: Region, b: Region, dilate: int = MATCH_DILATE_PX) -> tuple[float, int]:
    """(mask IoU, overlapping pixels after dilating both masks by `dilate` px)."""
    if not _boxes_near(a, b, dilate):
        return 0.0, 0
    x0, y0 = min(a.x0, b.x0) - dilate, min(a.y0, b.y0) - dilate
    x1, y1 = max(a.x1, b.x1) + dilate, max(a.y1, b.y1) + dilate
    ma, mb = a.window(x0, y0, x1, y1), b.window(x0, y0, x1, y1)
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    iou = inter / union if union else 0.0
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        da = cv2.dilate(ma.astype(np.uint8), k).astype(bool)
        db = cv2.dilate(mb.astype(np.uint8), k).astype(bool)
        near = int(np.logical_and(da, db).sum())
    else:
        near = inter
    return iou, near


def match(preds: list[Region], gts: list[Region], strict: bool = False,
          iou_thr: float | None = None, dilate: int = MATCH_DILATE_PX) -> MatchResult:
    iou_thr = (STRICT_IOU if strict else MATCH_IOU) if iou_thr is None else iou_thr
    order = sorted(range(len(preds)), key=lambda i: -preds[i].score)
    free = set(range(len(gts)))
    pairs, unmatched_pred = [], []
    for pi in order:
        best = None
        for gi in free:
            iou, near = pair_stats(preds[pi], gts[gi], 0 if strict else dilate)
            ok = iou >= iou_thr or (not strict and near > 0)
            if ok:
                key = (iou, near)
                if best is None or key > best[0]:
                    best = (key, gi)
        if best is None:
            unmatched_pred.append(pi)
        else:
            free.discard(best[1])
            pairs.append((pi, best[1], best[0][0]))
    return MatchResult(tp=len(pairs), fp=len(unmatched_pred), fn=len(free), pairs=pairs,
                       unmatched_pred=unmatched_pred, unmatched_gt=sorted(free))
