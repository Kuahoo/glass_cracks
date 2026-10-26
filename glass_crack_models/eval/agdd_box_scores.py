"""AGDD view gap (Phase 1): how strongly does each model respond inside the labeled boxes, per view?

    python -m glass_crack_models.eval.agdd_box_scores --runs E1_s0 E1_s1 E1_s2 E2_s0 E2_s1 E2_s2 E3_s0 --montage E1_s0

Reads the cached score maps (OUT_ROOT/evals/<run>/cache/agdd.npz) and the frozen threshold
(operating_point.json), so no GPU is needed. For every crack / scratches box (oriented box from labels/)
it records the highest score inside the box in `images/` and in `image/`, and per image the highest score
outside every labeled box. Writes to OUT_ROOT/results/:
- agdd_box_scores.csv: one row per run, box and view;
- agdd_box_scores.md: per run and view, mean / median in-box max, share of boxes reaching the frozen
  threshold and 0.1, how often `image/` scores below `images/` for the same box, mean max outside boxes;
- agdd_crack_crops_<run>.jpg (--montage): every crack box, `images/` crop next to the `image/` crop with the
  score heat map (GT box outline in green).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from glass_crack_models import paths
from glass_crack_models.eval import agdd as agdd_eval
from glass_crack_models.protocol import AGDD_NAMES, AGDD_VIEWS
from glass_crack_models.viz.overlay import GT_COLOR, caption, grid, read_rgb, save_rgb

GROUPS = {"crack": agdd_eval.CRACK_IDS, "linear": agdd_eval.LINEAR_IDS}


def obb_mask(obb: np.ndarray, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(obb).astype(np.int32)], 1)
    return m.astype(bool)


def box_mask(box, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x1, y1 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    m[y0:y1, x0:x1] = True
    return m


def gt_masks(img: agdd_eval.AgddImage, h: int, w: int) -> list[np.ndarray]:
    """Oriented-box masks at map resolution (axis-aligned box if the oriented one rasterizes to nothing)."""
    sx, sy = w / img.w, h / img.h
    out = []
    for obb, box in zip(img.obbs, img.boxes):
        m = obb_mask(obb * [sx, sy], h, w)
        out.append(m if m.any() else box_mask(box * [sx, sy, sx, sy], h, w))
    return out


def box_rows(run: str, images: list[agdd_eval.AgddImage], maps: Mapping[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    boxes, outside = [], []
    for img in images:
        for v in AGDD_VIEWS:
            key = f"{v}/{img.key}"
            if key not in maps:
                continue
            m = maps[key].astype(np.float32) / 255.0
            masks = gt_masks(img, *m.shape)
            for i, (c, gm) in enumerate(zip(img.classes, masks)):
                if c in agdd_eval.LINEAR_IDS:
                    boxes.append({"run": run, "image": img.key, "view": v, "box": i, "class": AGDD_NAMES[c],
                                  "in_box_max": float(m[gm].max()) if gm.any() else 0.0})
            covered = np.logical_or.reduce(masks) if masks else np.zeros(m.shape, bool)
            rest = m[~covered]
            outside.append({"run": run, "image": img.key, "view": v, "outside_max": float(rest.max()) if rest.size else 0.0})
    return boxes, outside


def summarize(boxes: list[dict], outside: list[dict], thr: float) -> list[dict]:
    rows = []
    for group, ids in GROUPS.items():
        names = {AGDD_NAMES[i] for i in ids}
        sel = [b for b in boxes if b["class"] in names]
        paired = {}
        for b in sel:
            paired.setdefault((b["image"], b["box"]), {})[b["view"]] = b["in_box_max"]
        both = [p for p in paired.values() if all(v in p for v in AGDD_VIEWS)]
        a, b = AGDD_VIEWS[0], AGDD_VIEWS[1]
        for v in AGDD_VIEWS:
            s = [x["in_box_max"] for x in sel if x["view"] == v]
            o = [x["outside_max"] for x in outside if x["view"] == v]
            rows.append({
                "boxes": group, "view": v, "n": len(s),
                "mean_in_box_max": statistics.fmean(s) if s else float("nan"),
                "median_in_box_max": statistics.median(s) if s else float("nan"),
                "share_at_thr": sum(x >= thr for x in s) / len(s) if s else float("nan"),
                "share_at_0.1": sum(x >= 0.1 for x in s) / len(s) if s else float("nan"),
                f"share_{b}_below_{a}": sum(p[b] < p[a] for p in both) / len(both) if both else float("nan"),
                "mean_outside_max": statistics.fmean(o) if o else float("nan"),
            })
    return rows


def heat(img: np.ndarray, score: np.ndarray) -> np.ndarray:
    h = cv2.applyColorMap(np.clip(score * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.55, cv2.cvtColor(h, cv2.COLOR_BGR2RGB), 0.45, 0)


def montage(run: str, images: list[agdd_eval.AgddImage], maps: Mapping[str, np.ndarray], out: Path,
            margin: int = 48, cell: int = 300):
    tiles = []
    for img in images:
        views = [v for v in AGDD_VIEWS if f"{v}/{img.key}" in maps and v in img.views]
        if not views:
            continue
        for i, c in enumerate(img.classes):
            if c not in agdd_eval.CRACK_IDS:
                continue
            for v in AGDD_VIEWS:
                key = f"{v}/{img.key}"
                if v not in views:
                    tiles.append(caption(np.full((cell, cell, 3), 255, np.uint8), f"{v}/ not cached"))
                    continue
                rgb = read_rgb(img.views[v])
                score = maps[key].astype(np.float32) / 255.0
                if score.shape != rgb.shape[:2]:
                    score = cv2.resize(score, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                vis = heat(rgb, score)
                cv2.polylines(vis, [np.round(img.obbs[i]).astype(np.int32)], True, GT_COLOR, 1)
                x0, y0, x1, y1 = img.boxes[i]
                h, w = rgb.shape[:2]
                ya, yb = max(0, int(y0) - margin), min(h, int(y1) + margin)
                xa, xb = max(0, int(x0) - margin), min(w, int(x1) + margin)
                m = obb_mask(img.obbs[i], h, w)
                peak = float(score[m].max()) if m.any() else 0.0
                crop = vis[ya:yb, xa:xb]
                s = cell / max(crop.shape[:2])
                crop = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
                                  interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                tiles.append(caption(crop, f"{v}/ max {peak:.2f} | {img.key}"))
    save_rgb(out, grid(tiles, cols=4, cell=cell + 22))


def operating_threshold(eval_dir: Path) -> float:
    f = eval_dir / "operating_point.json"
    return float(json.loads(f.read_text())["threshold"]) if f.exists() else float("nan")


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        wr.writeheader()
        wr.writerows(rows)


def _fmt(x) -> str:
    return "-" if isinstance(x, float) and x != x else (f"{x:.3f}" if isinstance(x, float) else str(x))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="Run folder names under OUT_ROOT/evals")
    ap.add_argument("--evals", type=Path, default=None, help="Default OUT_ROOT/evals")
    ap.add_argument("--montage", nargs="*", default=[], help="Runs to render crack-box crops for")
    ap.add_argument("--out", type=Path, default=None, help="Default OUT_ROOT/results")
    args = ap.parse_args(argv)

    evals = args.evals or paths.out_root() / "evals"
    out = args.out or paths.out_root() / "results"
    images = agdd_eval.load_agdd()
    all_boxes, md = [], ["# AGDD in-box scores per view\n",
                         "Highest cached score inside each labeled box (oriented box). `crack` = 29 crack boxes; "
                         "`linear` = crack + scratches. `share_image_below_images` compares the same box across views.\n"]
    for run in args.runs:
        cache = evals / run / "cache" / "agdd.npz"
        if not cache.exists():
            print(f"skip {run}: no {cache}")
            continue
        thr = operating_threshold(evals / run)
        with np.load(cache) as maps:
            boxes, outside = box_rows(run, images, maps)
            if run in args.montage:
                path = out / f"agdd_crack_crops_{run}.jpg"
                montage(run, images, maps, path)
                print(f"wrote {path}")
        all_boxes += boxes
        rows = summarize(boxes, outside, thr)
        cols = list(rows[0])
        md += [f"\n## {run} (frozen threshold {thr:.2f})\n", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        md += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    if not all_boxes:
        raise SystemExit("no AGDD caches found")
    write_csv(all_boxes, out / "agdd_box_scores.csv")
    text = "\n".join(md) + "\n"
    (out / "agdd_box_scores.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out / 'agdd_box_scores.csv'} and {out / 'agdd_box_scores.md'}")


if __name__ == "__main__":
    main()
