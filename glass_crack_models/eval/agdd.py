"""AGDD box-level evaluation (plan Section 4a)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from glass_crack_models import paths
from glass_crack_models.eval.regions import Region
from glass_crack_models.protocol import (AGDD_BOX_IOU, AGDD_HIT_FRACTION, AGDD_NONLINEAR_IDS, AGDD_STRICT_BOX_IOU,
                                         AGDD_TO_EVAL, AGDD_VIEWS)

LINEAR_IDS = AGDD_TO_EVAL["linear_defect"]
CRACK_IDS = AGDD_TO_EVAL["crack"]


@dataclass
class AgddImage:
    key: str                       # "<split>/<stem>"
    views: dict[str, Path]         # view folder -> image path
    classes: list[int]
    boxes: np.ndarray              # (n, 4) x0, y0, x1, y1 in pixels (from labels_rect/)
    obbs: list[np.ndarray]         # n x (4, 2) pixel polygons (from labels/)
    h: int
    w: int


def load_agdd(root: Path | None = None, limit: int | None = None) -> list[AgddImage]:
    root = root or paths.agdd_dir()
    out = []
    for split in ("train", "val"):
        for rect in sorted((root / "labels_rect" / split).glob("*.txt")):
            stem = rect.stem
            views = {}
            for v in AGDD_VIEWS:
                hits = [p for p in (root / v / split).glob(stem + ".*")]
                if hits:
                    views[v] = hits[0]
            if not views:
                continue
            h, w = cv2.imread(str(next(iter(views.values())))).shape[:2]
            classes, boxes, obbs = [], [], []
            rect_rows = [l.split() for l in rect.read_text().splitlines() if l.strip()]
            obb_rows = [l.split() for l in (root / "labels" / split / rect.name).read_text().splitlines() if l.strip()]
            if len(rect_rows) != len(obb_rows):
                raise ValueError(f"AGDD {split}/{stem}: labels_rect and labels disagree in length")
            for r, o in zip(rect_rows, obb_rows):
                c, cx, cy, bw, bh = int(r[0]), *map(float, r[1:5])
                classes.append(c)
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                obbs.append(np.array(o[1:9], float).reshape(4, 2) * [w, h])
            out.append(AgddImage(f"{split}/{stem}", views, classes, np.array(boxes, float).reshape(-1, 4), obbs, h, w))
            if limit and len(out) >= limit:
                return out
    return out


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _inside_fraction(pred: Region, obb: np.ndarray) -> float:
    shifted = np.round(obb - [pred.x0, pred.y0]).astype(np.int32)
    m = np.zeros(pred.mask.shape, np.uint8)
    cv2.fillPoly(m, [shifted], 1)
    return float(np.logical_and(pred.mask, m.astype(bool)).sum()) / max(pred.area, 1)


def _pred_box(r: Region):
    return (r.x0, r.y0, r.x1, r.y1)


def _greedy(preds, gt_ids, img: AgddImage, strict: bool):
    """Return {pred idx: gt idx} for one-to-one greedy matching by score."""
    free, out = set(gt_ids), {}
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i].score):
        pb, best = _pred_box(preds[pi]), None
        for gi in free:
            iou = box_iou(pb, img.boxes[gi])
            if strict:
                ok, key = iou >= AGDD_STRICT_BOX_IOU, (iou, 0.0)
            else:
                frac = _inside_fraction(preds[pi], img.obbs[gi])
                ok, key = iou >= AGDD_BOX_IOU or frac >= AGDD_HIT_FRACTION, (iou, frac)
            if ok and (best is None or key > best[0]):
                best = (key, gi)
        if best is not None:
            free.discard(best[1])
            out[pi] = best[1]
    return out


def eval_view(preds: list[Region], img: AgddImage) -> tuple[dict, set[int]]:
    """Counts for one image in one view, plus the set of matched GT box indices (for pair recall)."""
    gt_ids = [i for i, c in enumerate(img.classes) if c in LINEAR_IDS]
    nonlin = [i for i, c in enumerate(img.classes) if c in AGDD_NONLINEAR_IDS]
    lenient = _greedy(preds, gt_ids, img, strict=False)
    strict = _greedy(preds, gt_ids, img, strict=True)
    matched = set(lenient.values())
    confusions = 0
    for pi, p in enumerate(preds):
        if pi in lenient:
            continue
        pb = _pred_box(p)
        if any(box_iou(pb, img.boxes[g]) >= AGDD_BOX_IOU or _inside_fraction(p, img.obbs[g]) >= AGDD_HIT_FRACTION
               for g in nonlin):
            confusions += 1
    crack_gt = [i for i in gt_ids if img.classes[i] in CRACK_IDS]
    rec = {
        "images": 1, "gt": len(gt_ids), "pred": len(preds),
        "tp": len(lenient), "fp": len(preds) - len(lenient), "fn": len(gt_ids) - len(lenient),
        "tp_strict": len(strict), "fp_strict": len(preds) - len(strict), "fn_strict": len(gt_ids) - len(strict),
        "confusions": confusions,
        "gt_crack": len(crack_gt), "tp_crack": sum(1 for i in crack_gt if i in matched),
    }
    return rec, matched


def _div(a, b):
    return float(a) / float(b) if b else float("nan")


def pair_recall(s): return _div(s["pair_tp"], s["pair_gt"])
def crack_recall(s): return _div(s["tp_crack"], s["gt_crack"])
def confusions_per_100(s): return 100 * _div(s["confusions"], s["images"])
