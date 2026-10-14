"""Tile CUBIT-InSeg crack photos into a 640 px YOLO-seg pretraining set (plan Phase 4, step 3).

    python -m glass_crack_models.data.tile_cubit --out ${OUT_ROOT}/cubit_tiles                 # full run
    python -m glass_crack_models.data.tile_cubit --pilot 500 --out outputs/cubit_pilot           # 500 train images

Two passes per photo, 25% overlap each: 1024 px windows resized to 640, and native 640 px windows.
Polygons are clipped to each window, simplified (0.5 px at 640 scale) and fragments under 24 px² dropped.
Every crack tile is kept plus ~30% of crack-free tiles; tiles whose only crack pixels are dropped
fragments are skipped (neither positive nor a clean negative). CR1_* photos only (spalling skipped).
CUBIT's own train/val/test split is kept (val -> valid). Class 0 = linear_defect, so the head matches
the glass fine-tuning set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, box
from shapely.validation import make_valid

from glass_crack_models import paths
from glass_crack_models.data.rasterize import read_yolo
from glass_crack_models.infer.tiling import tile_starts
from glass_crack_models.protocol import MIN_REGION_PX

SPLITS = {"train": "train", "val": "valid", "test": "test"}
CRACK_ID = 0


def _polygons(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in _polygons(g)]
    return []


def _keep_negative(key: str, frac: float) -> bool:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < frac


def tile_image(job):
    img_path, out_split_dir, windows, out_size, overlap, neg_frac, quality, simplify = job
    t0 = time.time()
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    lbl = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    polys = []
    for c, p in read_yolo(lbl):
        if c != CRACK_ID or len(p) < 3:
            continue
        g = Polygon(p * [w, h])
        polys.extend(_polygons(g if g.is_valid else make_valid(g)))
    rows, nbytes = [], 0
    stats = {"pos": 0, "neg": 0, "neg_dropped": 0, "fragment_only": 0, "fragments_dropped": 0}
    for s in windows:
        scale = out_size / s
        for y0 in tile_starts(h, s, overlap):
            for x0 in tile_starts(w, s, overlap):
                win = box(x0, y0, x0 + s, y0 + s)
                lines, touched = [], False
                for p in polys:
                    if not p.intersects(win):
                        continue
                    for part in _polygons(p.intersection(win)):
                        touched = True
                        coords = (np.asarray(part.exterior.coords)[:-1] - [x0, y0]) * scale
                        q = Polygon(coords)
                        q = q.simplify(simplify, preserve_topology=True) if simplify > 0 else q
                        if q.is_empty or q.area < MIN_REGION_PX:
                            stats["fragments_dropped"] += 1
                            continue
                        for qq in _polygons(q):
                            xy = np.asarray(qq.exterior.coords)[:-1] / out_size
                            if len(xy) >= 3:
                                lines.append("0 " + " ".join(f"{v:.6f}" for v in np.clip(xy, 0, 1).ravel()))
                name = f"{img_path.stem}_w{s}_x{x0}_y{y0}"
                if not lines:
                    if touched:
                        stats["fragment_only"] += 1
                        continue
                    if not _keep_negative(name, neg_frac):
                        stats["neg_dropped"] += 1
                        continue
                crop = img[y0:y0 + s, x0:x0 + s]
                if crop.shape[0] != s or crop.shape[1] != s:
                    crop = cv2.copyMakeBorder(crop, 0, s - crop.shape[0], 0, s - crop.shape[1], cv2.BORDER_CONSTANT)
                if s != out_size:
                    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
                ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
                (out_split_dir / "images" / f"{name}.jpg").write_bytes(enc.tobytes())
                (out_split_dir / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                nbytes += len(enc)
                stats["pos" if lines else "neg"] += 1
                rows.append({"tile": f"{name}.jpg", "source_image": img_path.name, "window": s, "x0": x0, "y0": y0,
                             "n_polys": len(lines)})
    return {"image": img_path.name, "seconds": time.time() - t0, "bytes": nbytes, "rows": rows, **stats}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=None, help="Default: DATA_ROOT/datasets/cubit_inseg")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--pilot", type=int, default=0, help="Tile only N random train photos and report extrapolated totals")
    ap.add_argument("--windows", default="1024,640", help="Source window sizes in px (each resized to --out-size)")
    ap.add_argument("--out-size", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--neg-frac", type=float, default=0.3)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--simplify", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    src = args.src or paths.cubit_dir()
    out = paths.resolve(args.out)
    windows = [int(s) for s in args.windows.split(",")]
    splits = ["train"] if args.pilot else args.splits.split(",")
    all_cr = {sp: sorted((src / sp / "images").glob("CR*")) for sp in SPLITS}
    jobs = []
    for sp in splits:
        imgs = all_cr[sp]
        if args.pilot:
            imgs = sorted(random.Random(args.seed).sample(imgs, min(args.pilot, len(imgs))))
        d = out / SPLITS[sp]
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        jobs += [(p, d, windows, args.out_size, args.overlap, args.neg_frac, args.jpeg_quality, args.simplify) for p in imgs]

    t0 = time.time()
    res = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(tile_image, jobs, chunksize=2)):
            res.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} photos, {time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0

    split_of = {p.name: SPLITS[sp] for sp in splits for p in all_cr[sp]}
    with (out / "manifest.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["split", "tile", "source_image", "window", "x0", "y0", "n_polys"])
        wr.writeheader()
        for r in sorted(res, key=lambda r: r["image"]):
            for row in r["rows"]:
                wr.writerow({"split": split_of[r["image"]], **row})
    present = [SPLITS[s] for s in splits]
    (out / "data.yaml").write_text(
        f"# CUBIT-InSeg CR1 crack tiles ({args.windows} px windows -> {args.out_size} px), class 0 = linear_defect\n"
        f"path: {out}\ntrain: {'train/images' if 'train' in present else present[0] + '/images'}\n"
        f"val: {'valid/images' if 'valid' in present else present[0] + '/images'}\n"
        f"test: {'test/images' if 'test' in present else present[0] + '/images'}\n"
        "nc: 1\nnames: [linear_defect]\n")

    n = len(res)
    tot = {k: sum(r[k] for r in res) for k in ("pos", "neg", "neg_dropped", "fragment_only", "fragments_dropped", "bytes")}
    secs = [r["seconds"] for r in res]
    stats = {
        "photos": n, "windows": windows, "out_size": args.out_size, "overlap": args.overlap, "neg_frac": args.neg_frac,
        "jpeg_quality": args.jpeg_quality, "workers": args.workers, "wall_seconds": round(wall, 1),
        "seconds_per_photo_per_worker": round(float(np.mean(secs)), 2) if secs else None,
        "tiles_pos": tot["pos"], "tiles_neg": tot["neg"], "tiles_total": tot["pos"] + tot["neg"],
        "tiles_per_photo": round((tot["pos"] + tot["neg"]) / max(n, 1), 1),
        "neg_tiles_dropped": tot["neg_dropped"], "fragment_only_tiles_skipped": tot["fragment_only"],
        "fragments_dropped_lt_24px": tot["fragments_dropped"],
        "bytes": tot["bytes"], "mb": round(tot["bytes"] / 1e6, 1), "kb_per_tile": round(tot["bytes"] / 1e3 / max(1, tot["pos"] + tot["neg"]), 1),
    }
    if args.pilot:
        full = sum(len(v) for v in all_cr.values())
        f = full / max(n, 1)
        stats["extrapolated_full_cr_set"] = {
            "photos": full, "tiles": int(stats["tiles_total"] * f), "gb": round(tot["bytes"] * f / 1e9, 2),
            "wall_minutes_at_same_workers": round(wall * f / 60, 1),
        }
    (out / "tiling_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
