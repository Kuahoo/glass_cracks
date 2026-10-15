"""Per-image counts and summary statistics for the merged-set / NCUGD evaluation.

Every per-image record holds additive counts only, so any statistic below can be recomputed from
group sums (cluster bootstrap) or from a whole split.
"""
from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize

from glass_crack_models.eval.match import match
from glass_crack_models.eval.regions import Region, extract_regions, union_mask

SLICES = ("linear", "unannotated", "negdefect")  # has linear GT / no annotations / bubble-inclusion only


def image_record(score_map: np.ndarray, gts: list[Region], thr: float, slice_tag: str,
                 cldice: bool = True, preds: list[Region] | None = None, crack_label: int | None = None) -> dict:
    """`crack_label`: GT regions with this label are also counted separately (gt_crack / tp_crack)."""
    h, w = score_map.shape
    preds = extract_regions(score_map, thr) if preds is None else preds
    lenient, strict = match(preds, gts), match(preds, gts, strict=True)
    pm, gm = union_mask(preds, h, w), union_mask(gts, h, w)
    rec = {
        "images": 1, "gt": len(gts), "pred": len(preds),
        "tp": lenient.tp, "fp": lenient.fp, "fn": lenient.fn, "tp_pred": lenient.tp_pred,
        "tp_strict": strict.tp, "fp_strict": strict.fp, "fn_strict": strict.fn, "tp_pred_strict": strict.tp_pred,
        "inter": int(np.logical_and(pm, gm).sum()), "union": int(np.logical_or(pm, gm).sum()),
        f"images_{slice_tag}": 1, f"fp_{slice_tag}": lenient.fp,
    }
    if crack_label is not None:
        matched = {g for _, g, _ in lenient.pairs}
        rec["gt_crack"] = sum(1 for g in gts if g.label == crack_label)
        rec["tp_crack"] = sum(1 for i, g in enumerate(gts) if g.label == crack_label and i in matched)
    if cldice:
        sp = skeletonize(pm) if pm.any() else pm
        sg = skeletonize(gm) if gm.any() else gm
        rec.update(skel_p=int(sp.sum()), skel_p_in_g=int(np.logical_and(sp, gm).sum()),
                   skel_g=int(sg.sum()), skel_g_in_p=int(np.logical_and(sg, pm).sum()))
    return rec


def _div(a, b):
    return float(a) / float(b) if b else float("nan")


def recall(s): return _div(s["tp"], s["tp"] + s["fn"])
def precision(s):
    tp = s.get("tp_pred", s["tp"])
    return _div(tp, tp + s["fp"])


def strict_recall(s): return _div(s.get("tp_strict", 0), s.get("tp_strict", 0) + s.get("fn_strict", 0))


def strict_precision(s):
    tp = s.get("tp_pred_strict", s.get("tp_strict", 0))
    return _div(tp, tp + s.get("fp_strict", 0))
def mask_iou(s): return _div(s["inter"], s["union"])
def fp_per_100(s): return 100 * _div(s["fp"], s["images"])


def f1(s):
    p, r = precision(s), recall(s)
    if np.isnan(p) or np.isnan(r):
        return float("nan")
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def crack_recall(s): return _div(s.get("tp_crack", 0), s.get("gt_crack", 0))


def fp_per_100_slice(tag):
    return lambda s: 100 * _div(s.get(f"fp_{tag}", 0), s.get(f"images_{tag}", 0))


def cldice(s):
    tprec = _div(s.get("skel_p_in_g", 0), s.get("skel_p", 0))
    tsens = _div(s.get("skel_g_in_p", 0), s.get("skel_g", 0))
    if np.isnan(tprec) or np.isnan(tsens) or tprec + tsens == 0:
        return float("nan")
    return 2 * tprec * tsens / (tprec + tsens)


STATS = {
    "recall": recall, "precision": precision, "f1": f1, "mask_iou": mask_iou, "fp100": fp_per_100,
    "fp100_unannotated": fp_per_100_slice("unannotated"), "fp100_negdefect": fp_per_100_slice("negdefect"),
    "strict_recall": strict_recall, "strict_precision": strict_precision, "cldice": cldice,
    "crack_recall": crack_recall,
}
CI_STATS = ("recall", "precision", "mask_iou", "fp100", "fp100_unannotated", "fp100_negdefect")


def sum_records(records: list[dict]) -> dict:
    out: dict = {}
    for r in records:
        for k, v in r.items():
            out[k] = out.get(k, 0) + v
    return out


def summarize(records: list[dict]) -> dict:
    s = sum_records(records)
    out = {name: fn(s) for name, fn in STATS.items()}
    out.update(images=s.get("images", 0), gt=s.get("gt", 0), pred=s.get("pred", 0))
    return out
