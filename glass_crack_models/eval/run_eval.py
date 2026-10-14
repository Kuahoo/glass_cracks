"""Full evaluation suite for one checkpoint (plan Sections 4 and 4a).

    python -m glass_crack_models.eval.run_eval --checkpoint runs/E2_s0/weights/best.pt --exp-id E2 --seed 0

1. Predict `linear_defect` score maps on merged valid/test, NCUGD and both AGDD views (cached as uint8
   in <out>/cache/, so re-scoring never needs the GPU again: pass --reuse-cache).
2. Sweep thresholds on merged valid, pick max F1 (lenient matching), write operating_point.json next to
   the checkpoint. This threshold is then frozen for test and the external sets.
3. Valid and test at the frozen threshold with cluster-bootstrap CIs over near-duplicate groups;
   NCUGD; AGDD per view + pair recall; PR curves for every set; Track A mask mAP50.
4. Failure galleries on valid (top 20 false negatives / positives).
5. summary.json in <out>, one row appended to results.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from glass_crack_models import paths
from glass_crack_models.data.rasterize import image_paths, label_path_for, polygon_to_mask, read_yolo
from glass_crack_models.eval import agdd as agdd_eval
from glass_crack_models.eval import metrics as M
from glass_crack_models.eval.bootstrap import cluster_bootstrap
from glass_crack_models.eval.match import match
from glass_crack_models.eval.regions import Region, extract_regions, region_from_mask, union_mask
from glass_crack_models.infer.predictors import load_predictor
from glass_crack_models.infer.tiling import predict_tiled
from glass_crack_models.protocol import (BOOTSTRAP_RESAMPLES, LINEAR_SOURCE_IDS, NEGATIVE_DEFECT_IDS, OPERATING_POINT_RULE,
                                         THRESHOLD_GRID)
from glass_crack_models.viz.overlay import caption, grid, pred_overlay, read_rgb, save_rgb

CLS = "linear_defect"
CRACK_ID = 1  # in glass_merged_seg / ncugd class space


# ---- data ---------------------------------------------------------------------------------------

def spread(items: list, limit: int | None) -> list:
    """`limit` items evenly spaced over a sorted list (sorted order clusters by source dataset)."""
    if not limit or limit >= len(items):
        return items
    return [items[int(i * len(items) / limit)] for i in range(limit)]


def load_seg_split(split_dir: Path, groups: dict[str, str] | None = None, limit: int | None = None) -> list[dict]:
    """GT from 4-class labels: linear = crack + scratch; slice tag from what the image contains."""
    items = []
    for p in spread(image_paths(split_dir), limit):
        labels = read_yolo(label_path_for(p))
        h, w = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).shape[:2]
        gts = []
        for c, poly in labels:
            if c in LINEAR_SOURCE_IDS:
                r = region_from_mask(polygon_to_mask(poly, h, w), label=c)
                if r is not None:
                    gts.append(r)
        tag = "linear" if gts else ("negdefect" if any(c in NEGATIVE_DEFECT_IDS for c, _ in labels) else "unannotated")
        items.append({"name": p.name, "path": p, "h": h, "w": w, "gts": gts, "slice": tag,
                      "group": (groups or {}).get(p.name, p.name)})
    return items


def manifest_groups(ds: Path) -> dict[str, str]:
    with (ds / "manifest.csv").open(encoding="utf-8") as f:
        return {r["image"]: r["group"] for r in csv.DictReader(f)}


# ---- prediction cache -----------------------------------------------------------------------------

def predict_maps(predictor, image_list: list[tuple[str, Path]], cache: Path, ck_sha: str, reuse: bool) -> dict[str, np.ndarray]:
    meta = cache.with_suffix(".json")
    if reuse and cache.exists() and meta.exists() and json.loads(meta.read_text()).get("checkpoint_sha256") == ck_sha:
        with np.load(cache) as z:
            if set(z.files) >= {k for k, _ in image_list}:
                return {k: z[k] for k, _ in image_list}
    maps = {}
    for i, (key, path) in enumerate(image_list):
        m = predict_tiled(predictor, read_rgb(path))[CLS]
        maps[key] = np.clip(np.round(m * 255), 0, 255).astype(np.uint8)
        if (i + 1) % 50 == 0:
            print(f"  predicted {i + 1}/{len(image_list)} ({cache.stem})", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **maps)
    meta.write_text(json.dumps({"checkpoint_sha256": ck_sha, "n": len(maps), "time": time.ctime()}))
    return maps


def as_float(m: np.ndarray) -> np.ndarray:
    return m.astype(np.float32) / 255.0


# ---- scoring ------------------------------------------------------------------------------------------

def seg_records(items, maps, thr, cldice=True):
    return [M.image_record(as_float(maps[it["name"]]), it["gts"], thr, it["slice"], cldice=cldice, crack_label=CRACK_ID)
            for it in items]


def seg_summary(items, maps, thr, n_boot, with_ci=True):
    recs = seg_records(items, maps, thr)
    out = M.summarize(recs)
    if with_ci:
        groups = [it["group"] for it in items]
        for name in M.CI_STATS:
            _, lo, hi = cluster_bootstrap(recs, groups, M.STATS[name], n=n_boot)
            out[f"{name}_ci"] = [lo, hi]
    return out


def seg_sweep(items, maps):
    rows = []
    for thr in THRESHOLD_GRID:
        s = M.summarize(seg_records(items, maps, thr, cldice=False))
        rows.append({"thr": thr, **{k: s[k] for k in ("recall", "precision", "f1", "fp100")}})
    return rows


def pick_threshold(sweep_rows) -> dict:
    def key(r):
        f = r["f1"] if r["f1"] == r["f1"] else -1.0
        rec = r["recall"] if r["recall"] == r["recall"] else -1.0
        return (round(f, 6), rec)
    best = max(sweep_rows, key=key)
    if not best["f1"] == best["f1"] or best["f1"] <= 0:
        return {**best, "thr": 0.5, "note": "no valid F1 (model predicts nothing); fell back to 0.5"}
    return best


# ---- AGDD ---------------------------------------------------------------------------------------------

def agdd_records(images, maps, thr):
    per_view = {v: [] for v in agdd_eval.AGDD_VIEWS}
    pair = []
    for img in images:
        found = set()
        for v in agdd_eval.AGDD_VIEWS:
            key = f"{v}/{img.key}"
            if key not in maps:
                continue
            rec, matched = agdd_eval.eval_view(extract_regions(as_float(maps[key]), thr), img)
            per_view[v].append(rec)
            found |= matched
        gt_ids = [i for i, c in enumerate(img.classes) if c in agdd_eval.LINEAR_IDS]
        crack_ids = [i for i in gt_ids if img.classes[i] in agdd_eval.CRACK_IDS]
        pair.append({"pair_gt": len(gt_ids), "pair_tp": len(found & set(gt_ids)),
                     "pair_gt_crack": len(crack_ids), "pair_tp_crack": len(found & set(crack_ids))})
    return per_view, pair


def agdd_summary(images, maps, thr, n_boot, with_ci=True):
    per_view, pair = agdd_records(images, maps, thr)
    groups = [img.key for img in images]
    out = {}
    for v, recs in per_view.items():
        s = M.sum_records(recs)
        d = {"recall": M.recall(s), "precision": M.precision(s), "strict_recall": M.strict_recall(s),
             "fp100": M.fp_per_100(s), "confusions100": agdd_eval.confusions_per_100(s),
             "crack_recall": agdd_eval.crack_recall(s), "images": s.get("images", 0), "gt": s.get("gt", 0)}
        if with_ci and recs:
            _, lo, hi = cluster_bootstrap(recs, groups[:len(recs)], M.recall, n=n_boot)
            d["recall_ci"] = [lo, hi]
        out[v] = d
    s = M.sum_records(pair)
    out["pair"] = {"recall": agdd_eval.pair_recall(s),
                   "crack_recall": float(s["pair_tp_crack"]) / s["pair_gt_crack"] if s.get("pair_gt_crack") else float("nan"),
                   "gt": s.get("pair_gt", 0)}
    if with_ci and pair:
        _, lo, hi = cluster_bootstrap(pair, groups, agdd_eval.pair_recall, n=n_boot)
        out["pair"]["recall_ci"] = [lo, hi]
    return out


def agdd_sweep(images, maps):
    rows = []
    for thr in THRESHOLD_GRID:
        per_view, pair = agdd_records(images, maps, thr)
        row = {"thr": thr, "pair_recall": agdd_eval.pair_recall(M.sum_records(pair))}
        for v, recs in per_view.items():
            s = M.sum_records(recs)
            row[f"{v}_recall"], row[f"{v}_precision"] = M.recall(s), M.precision(s)
        rows.append(row)
    return rows


# ---- galleries, PR plot, bookkeeping --------------------------------------------------------------------

def _crop(img, r: Region, margin=48):
    h, w = img.shape[:2]
    x0, y0 = max(0, r.x0 - margin), max(0, r.y0 - margin)
    x1, y1 = min(w, r.x1 + margin), min(h, r.y1 + margin)
    return img[y0:y1, x0:x1]


def failure_galleries(items, maps, thr, out_dir: Path, top: int = 20):
    fns, fps = [], []
    for it in items:
        preds = extract_regions(as_float(maps[it["name"]]), thr)
        m = match(preds, it["gts"])
        fns += [(it["gts"][g].area, it, it["gts"][g]) for g in m.unmatched_gt]
        fps += [(preds[p].score, it, preds[p]) for p in m.unmatched_pred]
    for kind, lst in (("false_negatives", fns), ("false_positives", fps)):
        lst.sort(key=lambda t: -t[0])
        tiles = []
        for val, it, r in lst[:top]:
            img = read_rgb(it["path"])
            h, w = img.shape[:2]
            pm = union_mask(extract_regions(as_float(maps[it["name"]]), thr), h, w)
            gm = union_mask(it["gts"], h, w)
            ov = _crop(pred_overlay(img, pm, gm), r)
            label = f"area {val}" if kind == "false_negatives" else f"score {val:.2f}"
            tiles.append(caption(ov, f"{label} | {it['name'][:40]}"))
        save_rgb(out_dir / f"valid_{kind}.jpg", grid(tiles, cols=5, cell=256))


def plot_pr(curves: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (rows, rkey, pkey) in curves.items():
        r = [x[rkey] for x in rows]
        p = [x[pkey] for x in rows]
        ax.plot(r, p, marker=".", label=name)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_sha() -> str:
    h = hashlib.sha256()
    for f in (paths.merged_dir() / "manifest.csv", paths.data_root() / "processed" / "cleaning_report.json"):
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def git_commit() -> str:
    try:
        c = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True, text=True)
        d = subprocess.run(["git", "status", "--porcelain"], cwd=paths.REPO_ROOT, capture_output=True, text=True)
        return (c.stdout.strip() or "nogit") + ("-dirty" if d.stdout.strip() else "")
    except OSError:
        return "nogit"


def track_a_map50(checkpoint: Path, class_names: list[str], device: str) -> dict:
    from ultralytics import YOLO
    ds = paths.linear_dir() if class_names == ["linear_defect"] else paths.merged_dir()
    data = ds / "data_all.yaml"
    out = {}
    for split in ("val", "test"):
        r = YOLO(str(checkpoint)).val(data=str(data), split=split, imgsz=640, device=device, plots=False,
                                      verbose=False, project=str(paths.out_root() / "tmp_val"), name=split, exist_ok=True)
        out[split] = float(r.seg.map50)
    return out


def append_results(row: dict, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows, header = [], list(row)
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            rd = csv.DictReader(f)
            rows = list(rd)
            header = list(rd.fieldnames or []) + [k for k in row if k not in (rd.fieldnames or [])]
    rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in header})


def _fmt(x):
    return "" if x is None or (isinstance(x, float) and x != x) else (round(x, 4) if isinstance(x, float) else x)


def results_row(meta: dict, res: dict) -> dict:
    row = dict(meta)
    for split in ("valid", "test"):
        s = res.get(split)
        if not s:
            continue
        for k in ("recall", "precision", "f1", "mask_iou", "fp100", "fp100_unannotated", "fp100_negdefect",
                  "strict_recall", "strict_precision", "cldice", "crack_recall"):
            row[f"{split}_{k}"] = _fmt(s.get(k))
            if f"{k}_ci" in s:
                row[f"{split}_{k}_lo"], row[f"{split}_{k}_hi"] = (_fmt(v) for v in s[f"{k}_ci"])
    if res.get("map50"):
        row["valid_mask_map50"], row["test_mask_map50"] = _fmt(res["map50"]["val"]), _fmt(res["map50"]["test"])
    if res.get("ncugd"):
        s = res["ncugd"]
        for k in ("recall", "precision", "fp100", "strict_recall"):
            row[f"ncugd_{k}"] = _fmt(s.get(k))
        if "recall_ci" in s:
            row["ncugd_recall_lo"], row["ncugd_recall_hi"] = (_fmt(v) for v in s["recall_ci"])
    if res.get("agdd"):
        for v in agdd_eval.AGDD_VIEWS + ["pair"]:
            s = res["agdd"].get(v, {})
            for k in ("recall", "precision", "strict_recall", "fp100", "confusions100", "crack_recall"):
                if k in s:
                    row[f"agdd_{v}_{k}"] = _fmt(s[k])
            if "recall_ci" in s:
                row[f"agdd_{v}_recall_lo"], row[f"agdd_{v}_recall_hi"] = (_fmt(x) for x in s["recall_ci"])
    return row


# ---- main ---------------------------------------------------------------------------------------------

def run(args) -> dict:
    t0 = time.time()
    ck = args.checkpoint.resolve()
    out = args.out or paths.out_root() / "evals" / f"{args.exp_id}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    predictor = load_predictor(ck, args.device)
    ck_sha = file_sha256(ck)
    n_boot = args.bootstrap
    merged = paths.merged_dir()
    groups = manifest_groups(merged)
    res, curves = {}, {}

    print("[valid] predicting", flush=True)
    valid = load_seg_split(merged / "valid", groups, args.limit)
    vmaps = predict_maps(predictor, [(it["name"], it["path"]) for it in valid], out / "cache" / "valid.npz", ck_sha, args.reuse_cache)
    sweep = seg_sweep(valid, vmaps)
    op = pick_threshold(sweep)
    thr = float(op["thr"])
    curves["merged valid"] = (sweep, "recall", "precision")
    op_doc = {"threshold": thr, "rule": OPERATING_POINT_RULE, "valid_f1": op["f1"], "valid_recall": op["recall"],
              "valid_precision": op["precision"], "exp_id": args.exp_id, "seed": args.seed, "checkpoint": str(ck),
              "checkpoint_sha256": ck_sha, "note": op.get("note", ""), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "operating_point.json").write_text(json.dumps(op_doc, indent=1))
    if not args.limit:
        (ck.parent / "operating_point.json").write_text(json.dumps(op_doc, indent=1))
    print(f"[valid] frozen threshold {thr:.2f} (F1 {op['f1']:.3f})", flush=True)
    res["valid"] = seg_summary(valid, vmaps, thr, n_boot)
    failure_galleries(valid, vmaps, thr, out / "galleries")

    if not args.no_test:
        print("[test] predicting", flush=True)
        test = load_seg_split(merged / "test", groups, args.limit)
        tmaps = predict_maps(predictor, [(it["name"], it["path"]) for it in test], out / "cache" / "test.npz", ck_sha, args.reuse_cache)
        res["test"] = seg_summary(test, tmaps, thr, n_boot)
        curves["merged test"] = (seg_sweep(test, tmaps), "recall", "precision")

    if not args.skip_ncugd:
        print("[ncugd] predicting", flush=True)
        nc = load_seg_split(paths.ncugd_dir() / "test", None, args.limit)
        nmaps = predict_maps(predictor, [(it["name"], it["path"]) for it in nc], out / "cache" / "ncugd.npz", ck_sha, args.reuse_cache)
        s = seg_summary(nc, nmaps, thr, n_boot, with_ci=False)
        recs = seg_records(nc, nmaps, thr, cldice=False)
        _, lo, hi = cluster_bootstrap(recs, [it["name"] for it in nc], M.recall, n=n_boot)
        s["recall_ci"] = [lo, hi]
        res["ncugd"] = s
        curves["NCUGD"] = (seg_sweep(nc, nmaps), "recall", "precision")

    if not args.skip_agdd:
        print("[agdd] predicting", flush=True)
        images = spread(agdd_eval.load_agdd(), args.limit)
        lst = [(f"{v}/{img.key}", p) for img in images for v, p in img.views.items()]
        amaps = predict_maps(predictor, lst, out / "cache" / "agdd.npz", ck_sha, args.reuse_cache)
        res["agdd"] = agdd_summary(images, amaps, thr, n_boot)
        asw = agdd_sweep(images, amaps)
        for v in agdd_eval.AGDD_VIEWS:
            curves[f"AGDD {v}/ (box)"] = (asw, f"{v}_recall", f"{v}_precision")
        res["agdd_sweep"] = asw

    if predictor.track == "A" and not args.no_map and not args.limit:
        print("[map50] ultralytics val", flush=True)
        res["map50"] = track_a_map50(ck, predictor.class_names, predictor.device)

    res["sweeps"] = {k: v[0] for k, v in curves.items()}
    plot_pr(curves, out / "pr_curves.png")
    meta = {
        "exp_id": args.exp_id, "track": predictor.track, "task": args.task or ("yolo-seg" if predictor.track == "A" else "semseg"),
        "classes": "4class" if "crack" in predictor.class_names else "linear_defect",
        "pretrain": args.pretrain, "weighting": args.weighting, "seed": args.seed, "threshold": round(thr, 2),
        "checkpoint": str(ck), "checkpoint_sha256": ck_sha[:12], "git_commit": git_commit(), "dataset_sha": dataset_sha(),
        "limit": args.limit or "", "bootstrap": n_boot, "compute_units": args.compute_units or "",
        "eval_minutes": round((time.time() - t0) / 60, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": args.notes,
    }
    summary = {"meta": meta, "operating_point": op_doc, **res}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    row = results_row(meta, res)
    csv_path = args.results_csv or (out / "results_smoke.csv" if args.limit else paths.results_csv())
    append_results(row, csv_path)
    print_report(summary)
    print(f"wrote {out} and a row in {csv_path}")
    return summary


def print_report(s: dict):
    def ci(d, k):
        lo_hi = d.get(f"{k}_ci")
        return f"{d[k]:.3f}" + (f" [{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]" if lo_hi else "")
    for split in ("valid", "test"):
        if split in s:
            d = s[split]
            print(f"{split:6s} recall {ci(d, 'recall')}  precision {ci(d, 'precision')}  maskIoU {ci(d, 'mask_iou')}  "
                  f"FP/100 {ci(d, 'fp100')}  (unannotated {d['fp100_unannotated']:.1f}, bubble/incl {d['fp100_negdefect']:.1f})")
    if "ncugd" in s:
        print(f"NCUGD  recall {ci(s['ncugd'], 'recall')}  precision {s['ncugd']['precision']:.3f}")
    if "agdd" in s:
        a = s["agdd"]
        print("AGDD   " + "  ".join(f"{v}: recall {a[v]['recall']:.3f}" for v in agdd_eval.AGDD_VIEWS if v in a)
              + f"  pair recall {a['pair']['recall']:.3f}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default=None, help="e.g. yolo11s-seg, unet-resnet34 (for the results table)")
    ap.add_argument("--pretrain", default="")
    ap.add_argument("--weighting", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--compute-units", default="")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--results-csv", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    ap.add_argument("--limit", type=int, default=None, help="Smoke test: first N images per set (row goes to results_smoke.csv)")
    ap.add_argument("--reuse-cache", action="store_true", help="Re-score cached predictions (same checkpoint only)")
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--skip-ncugd", action="store_true")
    ap.add_argument("--skip-agdd", action="store_true")
    ap.add_argument("--no-map", action="store_true", help="Skip Ultralytics mask mAP50 (Track A)")
    return ap


def main(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
