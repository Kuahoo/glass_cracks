"""Crack prediction on arbitrary photos (plan Phase 3).

    python -m glass_crack_models.infer.predict --checkpoint best.pt --input photo.jpg --out preds/
    python -m glass_crack_models.infer.predict --checkpoint best.pth --input folder/ --out preds/ --scales 1.0,0.5

Per image: <stem>_overlay.jpg, <stem>_mask.png, <stem>_prob.npy (float16), <stem>.json (instances with
polygon, score, area and skeleton length in pixels). `run.json` records model, checkpoint and threshold.
The threshold defaults to the one frozen on merged valid (`operating_point.json` next to the checkpoint).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

from glass_crack_models.data.rasterize import IMAGE_EXTS, mask_to_polygons
from glass_crack_models.eval.regions import extract_regions, union_mask
from glass_crack_models.infer.predictors import load_predictor
from glass_crack_models.infer.tiling import auto_scales, predict_multiscale
from glass_crack_models.protocol import MIN_REGION_PX, TILE, TILE_OVERLAP
from glass_crack_models.viz.overlay import pred_overlay, read_rgb, save_rgb


def operating_point_for(checkpoint: Path) -> dict | None:
    f = checkpoint.parent / "operating_point.json"
    return json.loads(f.read_text()) if f.exists() else None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def instances_json(score_map: np.ndarray, thr: float) -> list[dict]:
    out = []
    for r in extract_regions(score_map, thr, MIN_REGION_PX):
        polys = mask_to_polygons(r.mask, epsilon=1.0)
        out.append({
            "score": round(r.score, 4),
            "area_px": r.area,
            "length_px": int(skeletonize(r.mask).sum()),
            "bbox_xyxy": [r.x0, r.y0, r.x1, r.y1],
            "polygons": [(p + [r.x0, r.y0]).round(1).tolist() for p in polys],
        })
    return sorted(out, key=lambda d: -d["score"])


def predict_one(predictor, img_path: Path, out_dir: Path, thr: float, scales, cls: str, tile: int, overlap: float):
    img = read_rgb(img_path)
    h, w = img.shape[:2]
    sc = auto_scales(h, w, tile) if scales == "auto" else scales
    t0 = time.time()
    maps = predict_multiscale(predictor, img, sc, thr=thr, tile=tile, overlap=overlap)
    score = maps[cls]
    inst = instances_json(score, thr)
    mask = union_mask(extract_regions(score, thr, MIN_REGION_PX), h, w)
    stem = img_path.stem
    save_rgb(out_dir / f"{stem}_overlay.jpg", pred_overlay(img, mask))
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask.astype(np.uint8) * 255)
    np.save(out_dir / f"{stem}_prob.npy", score.astype(np.float16))
    (out_dir / f"{stem}.json").write_text(json.dumps({
        "image": str(img_path), "height": h, "width": w, "class": cls, "threshold": thr, "scales": sc,
        "seconds": round(time.time() - t0, 2), "n_instances": len(inst), "instances": inst}, separators=(",", ":")))
    return len(inst), time.time() - t0


def export(predictor, fmt: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if predictor.track == "A":
        print("exported", predictor.model.export(format=fmt, imgsz=TILE))
        return
    import torch
    x = torch.zeros(1, 3, TILE, TILE, device=predictor.device)
    model = predictor.model.eval()
    if fmt == "torchscript":
        path = out_dir / (Path(predictor.checkpoint).stem + ".torchscript")
        torch.jit.trace(model, x).save(str(path))
    else:
        path = out_dir / (Path(predictor.checkpoint).stem + ".onnx")
        torch.onnx.export(model, x, str(path), input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "b", 2: "h", 3: "w"}, "logits": {0: "b", 2: "h", 3: "w"}})
    print("exported", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--input", type=Path, help="Image file or folder")
    ap.add_argument("--out", type=Path, default=Path("predictions"))
    ap.add_argument("--threshold", type=float, default=None, help="Default: operating_point.json next to checkpoint")
    ap.add_argument("--scales", default="auto", help="'auto' or comma list relative to native size, e.g. 1.0,0.5")
    ap.add_argument("--class", dest="cls", default="linear_defect")
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--overlap", type=float, default=TILE_OVERLAP)
    ap.add_argument("--device", default=None)
    ap.add_argument("--export", choices=["torchscript", "onnx"], default=None, help="Export the model and exit")
    args = ap.parse_args(argv)

    predictor = load_predictor(args.checkpoint, args.device)
    if args.export:
        export(predictor, args.export, args.out)
        return
    if args.input is None:
        ap.error("--input is required unless --export is given")
    op = operating_point_for(args.checkpoint)
    if args.threshold is not None:
        thr, thr_src = args.threshold, "cli"
    elif op is not None:
        thr, thr_src = float(op["threshold"]), str(args.checkpoint.parent / "operating_point.json")
    else:
        ap.error("no operating_point.json next to the checkpoint; run run_eval first or pass --threshold")
    scales = "auto" if args.scales == "auto" else [float(s) for s in args.scales.split(",")]
    files = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint),
        "track": predictor.track, "model_classes": predictor.class_names, "class": args.cls,
        "threshold": thr, "threshold_source": thr_src, "scales": args.scales, "tile": args.tile,
        "overlap": args.overlap, "min_region_px": MIN_REGION_PX, "device": predictor.device,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_images": len(files)}, indent=1))
    for f in files:
        n, sec = predict_one(predictor, f, args.out, thr, scales, args.cls, args.tile, args.overlap)
        print(f"{f.name}: {n} instances, {sec:.1f}s")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
