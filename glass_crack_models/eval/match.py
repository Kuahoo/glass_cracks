"""Fixed instance-matching rule (plan Section 4).

Lenient (headline): a prediction matches a GT instance if mask IoU >= 0.3, or if the two masks
overlap after both are dilated by 3 px. Strict: mask IoU >= 0.5 only.
Matching is greedy, predictions taken in descending score order. Each GT is matched at most once.
A prediction normally claims one GT; it claims every free GT it matches when its IoU with the union of
those GTs reaches the same IoU threshold (one connected prediction over touching GT instances).
`tp` counts matched GTs (recall), `tp_pred` counts matched predictions (precision).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from glass_crack_models.eval.regions import Region
from glass_crack_models.protocol import MATCH_DILATE_PX, MATCH_IOU, STRICT_IOU


@dataclass
class MatchResult:
    tp: int  # matched GTs
    fp: int
    fn: int
    tp_pred: int = 0  # matched predictions
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


def union_iou(pred: Region, gts: list[Region]) -> float:
    x0, y0 = min([pred.x0] + [g.x0 for g in gts]), min([pred.y0] + [g.y0 for g in gts])
    x1, y1 = max([pred.x1] + [g.x1 for g in gts]), max([pred.y1] + [g.y1 for g in gts])
    pm = pred.window(x0, y0, x1, y1)
    gm = np.zeros_like(pm)
    for g in gts:
        gm |= g.window(x0, y0, x1, y1)
    union = int(np.logical_or(pm, gm).sum())
    return int(np.logical_and(pm, gm).sum()) / union if union else 0.0


def match(preds: list[Region], gts: list[Region], strict: bool = False,
          iou_thr: float | None = None, dilate: int = MATCH_DILATE_PX) -> MatchResult:
    iou_thr = (STRICT_IOU if strict else MATCH_IOU) if iou_thr is None else iou_thr
    order = sorted(range(len(preds)), key=lambda i: -preds[i].score)
    free = set(range(len(gts)))
    pairs, unmatched_pred, tp_pred = [], [], 0
    for pi in order:
        pred = preds[pi]
        touching = {}
        for gi in sorted(free):
            iou, near = pair_stats(pred, gts[gi], 0 if strict else dilate)
            if near > 0:
                touching[gi] = (iou, near)
        claimed = []
        if len(touching) > 1:
            # Grow a GT group while the prediction's IoU with the group union does not drop
            # (ties admit GTs nested inside already-claimed ones).
            group = [max(touching, key=touching.get)]
            cur = touching[group[0]][0]
            rest = [gi for gi in touching if gi != group[0]]
            while rest:
                val, gi = max((union_iou(pred, [gts[g] for g in group + [c]]), c) for c in rest)
                if val < cur:
                    break
                group.append(gi)
                rest.remove(gi)
                cur = val
            if len(group) > 1 and cur >= iou_thr:
                claimed = group
        if not claimed:
            ok = [gi for gi, (iou, near) in touching.items() if iou >= iou_thr or not strict]
            if ok:
                claimed = [max(ok, key=touching.get)]
        if not claimed:
            unmatched_pred.append(pi)
            continue
        for gi in claimed:
            free.discard(gi)
            pairs.append((pi, gi, touching[gi][0]))
        tp_pred += 1
    return MatchResult(tp=len(pairs), fp=len(unmatched_pred), fn=len(free), tp_pred=tp_pred, pairs=pairs,
                       unmatched_pred=unmatched_pred, unmatched_gt=sorted(free))
