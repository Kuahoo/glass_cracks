#!/usr/bin/env python3
"""check_ncugd_pipeline.py - find out WHY a model scores recall 0.0 on an external test set (NCUGD).

Three questions, answered separately so you know what to fix:

  A. Is the ground truth where the crack is?   (label / coordinate bug)
     Rasterizes the GT polygons exactly as a YOLO-seg loader would, draws overlays, and scores several
     "what if the coordinates were mis-handled" hypotheses (x/y swapped, flipped, W/H mixed up). A crack
     is a visible line, so the correct placement should show clearly more local contrast than the wrong
     ones and than randomly shifted copies of the same polygons.

  B. Does the model respond at all?            (domain / scale gap)
     Runs the model at several input sizes and resize modes (aspect-preserving letterbox vs stretch)
     and records the probability INSIDE each GT instance vs the highest probability anywhere else.

  C. Is the threshold killing it?              (calibration)
     Instance recall at a sweep of thresholds, using the matching rule from the evaluation protocol:
     hit = IoU >= 0.3 with a predicted blob OR any overlap after dilating the prediction by 3 px
     (strict IoU >= 0.5 is reported too).

Outputs (default ./ncugd_check/): report.md, results.json, instances.csv, gt_overlay/*.jpg,
maps/<size>_<mode>/*.jpg (image | probability overlay, GT outline in green).

Backends (model is only needed for B and C; A runs with --backend none):
  ultralytics  --weights best.pt [--model-classes 1 3]       YOLO-seg; prob = max(conf * instance mask)
  smp          --weights unet.pth --encoder resnet34         segmentation_models_pytorch; sigmoid output
  custom       --module my_infer.py                          def predict(image_bgr, size, mode) -> HxW float [0,1]
                                                             (any subset of the arguments works)
  pred-dir     --pred-dir DIR                                precomputed <stem>.npy (float) or <stem>.png (8/16-bit)
  none                                                       GT / alignment check only

Examples
  python check_ncugd_pipeline.py --data glass_crack_dataset_bundle/processed/ncugd_external_test --backend none
  python check_ncugd_pipeline.py --data .../ncugd_external_test --backend ultralytics \\
         --weights runs/segment/e1/weights/best.pt --sizes 640 1024 1280 --op-threshold 0.25
  python check_ncugd_pipeline.py --data .../ncugd_external_test --backend smp --weights unet_best.pth \\
         --encoder resnet34 --op-threshold 0.95
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("opencv-python is required: pip install opencv-python")

SCRIPT_VERSION = "1.0"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_THRESHOLDS = [0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
GREEN = (0, 255, 0)


# ================================================================== ground truth loading
def find_pairs(data: Path, images_dir: Path | None = None, labels_dir: Path | None = None):
    """Pair images with YOLO label files by stem. Returns (list of (stem, img_path, label_path|None), duplicate stems).

    By default everything under `data` is searched. Give images_dir / labels_dir (absolute, or relative to data)
    to pick one folder each, e.g. AGDD's two views: --images-dir data/images  vs  --images-dir data/image.
    """
    def resolve(d, default):
        if d is None:
            return default
        return d if d.is_absolute() else data / d
    img_root, lab_root = resolve(images_dir, data), resolve(labels_dir, data)
    imgs, txts, dup = {}, {}, []
    for p in sorted(img_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            # masks/overlays folders are not source images
            if any(part.lower() in ("masks", "mask", "overlay", "overlays", "gt_overlay") for part in p.relative_to(img_root).parts):
                continue
            if p.stem in imgs:
                dup.append(p.stem)
                continue
            imgs[p.stem] = p
    for p in sorted(lab_root.rglob("*.txt")):
        if p.name.lower() not in ("readme.txt", "classes.txt", "labels.txt"):
            txts.setdefault(p.stem, p)
    return [(s, imgs[s], txts.get(s)) for s in sorted(imgs)], dup


def parse_label(path: Path | None, keep_classes=None):
    """YOLO-seg lines -> list of dict(cls, pts Nx2 normalized). Boxes become rectangles."""
    out, skipped = [], 0
    if path is None:
        return out, skipped
    for ln in path.read_text(errors="ignore").splitlines():
        tk = ln.split()
        if not tk:
            continue
        try:
            v = [float(t) for t in tk]
        except ValueError:
            skipped += 1
            continue
        cls, rest = int(v[0]), v[1:]
        if keep_classes is not None and cls not in keep_classes:
            continue
        if len(rest) == 4:  # box: cx cy w h
            cx, cy, w, h = rest
            pts = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                            [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
        elif len(rest) >= 6 and len(rest) % 2 == 0:
            pts = np.array(rest).reshape(-1, 2)
        else:
            skipped += 1
            continue
        out.append({"cls": cls, "pts": pts})
    return out, skipped


HYPOTHESES = {
    "normal": lambda x, y, W, H: (x * W, y * H),
    "swap_xy": lambda x, y, W, H: (y * W, x * H),
    "flip_x": lambda x, y, W, H: ((1 - x) * W, y * H),
    "flip_y": lambda x, y, W, H: (x * W, (1 - y) * H),
    "wh_mixed_up": lambda x, y, W, H: (x * H, y * W),
    "square_scale": lambda x, y, W, H: (x * max(W, H), y * max(W, H)),
}


def to_pixels(pts, W, H, hyp="normal"):
    fx, fy = HYPOTHESES[hyp](pts[:, 0], pts[:, 1], W, H)
    return np.stack([fx, fy], 1)


def raster(pts_px, W, H):
    m = np.zeros((H, W), np.uint8)
    p = np.round(pts_px).astype(np.int32)
    if len(p) >= 3:
        cv2.fillPoly(m, [p], 1)
    return m.astype(bool)


# ================================================================== alignment check (A)
def contrast_score(gray, mask):
    if mask.sum() < 5:
        return None
    m8 = mask.astype(np.uint8)
    outer = cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    inner = cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ring = (outer > 0) & (inner == 0)
    if ring.sum() < 10:
        return None
    g = gray.astype(np.float32)
    return float(abs(g[mask].mean() - g[ring].mean()) / (g[ring].std() + 2.0))


def alignment_check(samples, seed=0, n_shift=10):
    rng = np.random.default_rng(seed)
    scores = {h: [] for h in HYPOTHESES}
    chance = []
    for s in samples:
        img = s["img"]
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for inst in s["insts"]:
            for h in HYPOTHESES:
                if h in ("wh_mixed_up", "square_scale") and W == H:
                    continue
                sc = contrast_score(gray, raster(to_pixels(inst["pts"], W, H, h), W, H))
                if sc is not None:
                    scores[h].append(sc)
            base = raster(to_pixels(inst["pts"], W, H, "normal"), W, H).astype(np.uint8)
            for _ in range(n_shift):
                dx, dy = rng.uniform(-0.3, 0.3) * W, rng.uniform(-0.3, 0.3) * H
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                sh = cv2.warpAffine(base, M, (W, H), flags=cv2.INTER_NEAREST).astype(bool)
                sc = contrast_score(gray, sh)
                if sc is not None:
                    chance.append(sc)
    res = {h: {"mean": round(float(np.mean(v)), 3), "n": len(v)} for h, v in scores.items() if v}
    ch = float(np.mean(chance)) if chance else float("nan")
    return res, round(ch, 3)


# ================================================================== backends (B)
def _letterbox(img, size, pad_value=114, multiple=32):
    H, W = img.shape[:2]
    S = int(np.ceil(size / multiple) * multiple)
    s = S / max(H, W)
    nh, nw = int(round(H * s)), int(round(W * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.full((S, S, 3), pad_value, np.uint8)
    top, left = (S - nh) // 2, (S - nw) // 2
    canvas[top:top + nh, left:left + nw] = r
    return canvas, (top, left, nh, nw, S)


def _unletterbox(prob, meta, H, W):
    top, left, nh, nw, _ = meta
    crop = prob[top:top + nh, left:left + nw]
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)


def make_backend(args):
    b = args.backend
    if b == "none":
        return None
    if b == "pred-dir":
        d = Path(args.pred_dir)

        def predict(img, size, mode, stem):
            H, W = img.shape[:2]
            for ext in (".npy", ".png", ".tif", ".tiff"):
                f = d / f"{stem}{ext}"
                if f.exists():
                    if ext == ".npy":
                        a = np.load(f).astype(np.float32)
                        if a.ndim == 3:
                            a = a.squeeze()
                    else:
                        raw = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                        if raw.ndim == 3:
                            raw = raw[..., 0]
                        a = raw.astype(np.float32) / (65535.0 if raw.dtype == np.uint16 else 255.0)
                    if a.shape != (H, W):
                        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
                    return a
            raise FileNotFoundError(f"no prediction for '{stem}' in {d}")
        predict.configs = [("external", "as-is")]
        return predict
    if b == "custom":
        spec = importlib.util.spec_from_file_location("user_infer", args.module)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = mod.predict
        params = set(inspect.signature(fn).parameters)

        def predict(img, size, mode, stem):
            kw = {k: v for k, v in (("size", size), ("mode", mode), ("stem", stem)) if k in params}
            return np.asarray(fn(img, **kw), dtype=np.float32)
        return predict
    if b == "ultralytics":
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("ultralytics is not installed: pip install ultralytics")
        model = YOLO(args.weights)
        keep = set(args.model_classes) if args.model_classes else None

        def predict(img, size, mode, stem):
            H, W = img.shape[:2]
            inp = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR) if mode == "stretch" else img
            res = model.predict(inp, imgsz=size, conf=0.001, iou=0.7, max_det=300,
                                device=args.device, verbose=False)[0]
            ph, pw = inp.shape[:2]
            prob = np.zeros((ph, pw), np.float32)
            if res.masks is not None and res.boxes is not None:
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                for xy, c, k in zip(res.masks.xy, confs, clss):
                    if keep is not None and int(k) not in keep:
                        continue
                    if len(xy) < 3:
                        continue
                    tmp = np.zeros((ph, pw), np.uint8)
                    cv2.fillPoly(tmp, [np.round(xy).astype(np.int32)], 1)
                    prob = np.maximum(prob, tmp.astype(np.float32) * float(c))
            if mode == "stretch":
                prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
            return prob
        return predict
    if b == "smp":
        try:
            import torch
            import segmentation_models_pytorch as smp
        except ImportError:
            sys.exit("torch / segmentation-models-pytorch not installed: pip install torch segmentation-models-pytorch")
        dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            ckpt = torch.load(args.weights, map_location=dev, weights_only=True)
        except Exception:
            print("  note: weights_only load failed, retrying with full unpickling (only do this for your own files)")
            ckpt = torch.load(args.weights, map_location=dev, weights_only=False)
        if isinstance(ckpt, torch.nn.Module):
            net = ckpt
        else:
            state = ckpt
            for k in ("model_state_dict", "state_dict", "model"):
                if isinstance(state, dict) and k in state and isinstance(state[k], dict):
                    state = state[k]
                    break
            net = smp.create_model(args.arch, encoder_name=args.encoder, encoder_weights=None,
                                   in_channels=3, classes=args.smp_classes)
            try:
                net.load_state_dict(state, strict=True)
            except RuntimeError:
                stripped = {k.split("module.", 1)[-1].split("model.", 1)[-1] if k.startswith(("module.", "model.")) else k: v
                            for k, v in state.items()}
                missing = net.load_state_dict(stripped, strict=False)
                n_total = len(net.state_dict())
                if len(missing.missing_keys) > 0.05 * n_total:
                    sys.exit(f"checkpoint does not match --arch {args.arch} / --encoder {args.encoder}: "
                             f"{len(missing.missing_keys)}/{n_total} keys missing")
                print(f"  note: loaded with {len(missing.missing_keys)} missing / {len(missing.unexpected_keys)} unexpected keys")
        net.to(dev).eval()
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)

        def predict(img, size, mode, stem):
            H, W = img.shape[:2]
            if mode == "stretch":
                S = int(np.ceil(size / 32) * 32)
                inp, meta = cv2.resize(img, (S, S), interpolation=cv2.INTER_LINEAR), None
            else:
                inp, meta = _letterbox(img, size)
            x = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if args.normalize == "imagenet":
                x = (x - mean) / std
            t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dev)
            with torch.no_grad():
                out = net(t)
                if args.smp_classes == 1:
                    p = torch.sigmoid(out)[0, 0]
                else:
                    p = torch.softmax(out, 1)[0, args.smp_channel]
            p = p.cpu().numpy().astype(np.float32)
            if meta is None:
                return cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)
            return _unletterbox(p, meta, H, W)
        return predict
    raise SystemExit(f"unknown backend {b}")


# ================================================================== metrics (C)
def instance_metrics(prob, gt_masks, thresholds, dil_radius=3):
    k = 2 * dil_radius + 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = {}
    for t in thresholds:
        pred8 = (prob >= t).astype(np.uint8)
        n, lab = cv2.connectedComponents(pred8, connectivity=8)
        areas = np.bincount(lab.ravel(), minlength=n)
        pred_d = cv2.dilate(pred8, kern).astype(bool)
        rows = []
        for g in gt_masks:
            ga = int(g.sum())
            if n > 1 and ga:
                inter = np.bincount(lab[g], minlength=n)
                iou = inter[1:] / np.maximum(areas[1:] + ga - inter[1:], 1)
                best = float(iou.max())
            else:
                best = 0.0
            rows.append({"best_iou": best, "hit_dil": bool((pred_d & g).any()),
                         "pix_recall": float(pred8[g].mean()) if ga else 0.0})
        out[t] = rows
    return out


# ================================================================== visuals
def draw_gt(img, gt_masks, color=GREEN):
    out = img.copy()
    for g in gt_masks:
        cs, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, color, 1)
    return out


def heat_overlay(img, prob, gt_masks, text=""):
    h = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blend = cv2.addWeighted(img, 0.5, h, 0.5, 0)
    blend = draw_gt(blend, gt_masks)
    if text:
        cv2.putText(blend, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(blend, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return blend


# ================================================================== main
def pct_str(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}%"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder containing NCUGD images + YOLO-seg label txt files")
    ap.add_argument("--images-dir", type=Path, help="only use images from this folder (relative to --data or absolute)")
    ap.add_argument("--labels-dir", type=Path, help="only use label txt files from this folder (relative to --data or absolute)")
    ap.add_argument("--backend", choices=["none", "ultralytics", "smp", "custom", "pred-dir"], default="none")
    ap.add_argument("--weights", type=str, help="model weights (ultralytics / smp)")
    ap.add_argument("--module", type=str, help="python file defining predict(image_bgr, [size], [mode], [stem])")
    ap.add_argument("--pred-dir", type=str, help="folder of precomputed probability maps")
    ap.add_argument("--arch", default="unet", help="smp architecture (default unet)")
    ap.add_argument("--encoder", default="resnet34", help="smp encoder (default resnet34)")
    ap.add_argument("--smp-classes", type=int, default=1)
    ap.add_argument("--smp-channel", type=int, default=0, help="channel used when --smp-classes > 1")
    ap.add_argument("--normalize", choices=["imagenet", "none"], default="imagenet", help="smp input normalization")
    ap.add_argument("--model-classes", type=int, nargs="*", help="ultralytics: only use these predicted class ids")
    ap.add_argument("--gt-classes", type=int, nargs="*", help="only score GT lines with these class ids (default all)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 1024, 1280])
    ap.add_argument("--modes", nargs="+", choices=["letterbox", "stretch"], default=["letterbox", "stretch"])
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--op-threshold", type=float, default=0.5, help="your model's operating threshold (e.g. 0.95 for the U-Net)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only first N images")
    ap.add_argument("--max-vis", type=int, default=40, help="max images to render per configuration")
    ap.add_argument("--out", type=Path, default=Path("ncugd_check"))
    args = ap.parse_args(argv)

    if args.backend in ("ultralytics", "smp") and not args.weights:
        ap.error("--weights is required for this backend")
    if args.backend == "custom" and not args.module:
        ap.error("--module is required for --backend custom")
    if args.backend == "pred-dir" and not args.pred_dir:
        ap.error("--pred-dir is required for --backend pred-dir")
    if not args.data.is_dir():
        ap.error(f"not a directory: {args.data}")

    out = args.out
    (out / "gt_overlay").mkdir(parents=True, exist_ok=True)
    thresholds = sorted(set(args.thresholds) | {args.op_threshold})

    # ---------------------------------------------------------------- load GT
    pairs, dup_stems = find_pairs(args.data, args.images_dir, args.labels_dir)
    if args.limit:
        pairs = pairs[:args.limit]
    keep = set(args.gt_classes) if args.gt_classes else None
    samples, skipped_lines, no_label, unreadable = [], 0, [], []
    for stem, ip, lp in pairs:
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img is None:
            unreadable.append(str(ip))
            continue
        insts, sk = parse_label(lp, keep)
        skipped_lines += sk
        if lp is None:
            no_label.append(stem)
        samples.append({"stem": stem, "img": img, "insts": insts, "img_path": ip})
    n_inst = sum(len(s["insts"]) for s in samples)
    if not samples:
        sys.exit(f"no readable images under {args.data}")
    print(f"loaded {len(samples)} images, {n_inst} GT instances from {args.data}")

    sizes = {}
    for s in samples:
        h, w = s["img"].shape[:2]
        sizes[(w, h)] = sizes.get((w, h), 0) + 1
    all_pts = np.concatenate([i["pts"] for s in samples for i in s["insts"]]) if n_inst else np.zeros((0, 2))
    sanity = {
        "images": len(samples), "instances": n_inst,
        "image_sizes": {f"{w}x{h}": n for (w, h), n in sizes.items()},
        "images_without_label_file": len(no_label), "images_with_zero_instances": sum(1 for s in samples if not s["insts"]),
        "skipped_label_lines": skipped_lines, "duplicate_stems_ignored": len(dup_stems), "unreadable": unreadable,
        "coord_min": float(all_pts.min()) if len(all_pts) else None,
        "coord_max": float(all_pts.max()) if len(all_pts) else None,
        "coords_outside_unit_square": int(((all_pts < -1e-6) | (all_pts > 1 + 1e-6)).any(axis=1).sum()) if len(all_pts) else 0,
    }
    # GT size stats (normal hypothesis)
    widths, areas = [], []
    for s in samples:
        H, W = s["img"].shape[:2]
        for i in s["insts"]:
            px = to_pixels(i["pts"], W, H)
            a = 0.5 * abs(np.dot(px[:, 0], np.roll(px[:, 1], -1)) - np.dot(px[:, 1], np.roll(px[:, 0], -1)))
            per = float(np.sum(np.hypot(*(px - np.roll(px, -1, 0)).T)))
            areas.append(a)
            if per > 0 and a > 0:
                widths.append(2 * a / per)
    sanity["gt_area_px_median"] = round(float(np.median(areas)), 1) if areas else None
    sanity["gt_stroke_width_px_median"] = round(float(np.median(widths)), 2) if widths else None

    # ---------------------------------------------------------------- A: alignment
    align, chance = alignment_check(samples)
    best_h = max(align, key=lambda h: align[h]["mean"]) if align else None
    normal = align.get("normal", {}).get("mean")
    for s in samples[:args.max_vis]:
        H, W = s["img"].shape[:2]
        gts = [raster(to_pixels(i["pts"], W, H, "normal"), W, H) for i in s["insts"]]
        panel = [draw_gt(s["img"], gts)]
        if best_h and best_h != "normal":
            alt = [raster(to_pixels(i["pts"], W, H, best_h), W, H) for i in s["insts"]]
            panel.append(draw_gt(s["img"], alt, (0, 0, 255)))
        cv2.imwrite(str(out / "gt_overlay" / f"{s['stem']}.jpg"), np.hstack(panel))

    verdicts = []
    if align and normal is not None and chance == chance:
        if best_h != "normal" and align[best_h]["mean"] > 1.5 * normal and align[best_h]["mean"] > 2 * chance:
            verdicts.append(f"**A. LABEL / COORDINATE PROBLEM LIKELY.** Hypothesis `{best_h}` places the GT on visible structure "
                            f"({align[best_h]['mean']}) far better than the loader's normal placement ({normal}); chance level is {chance}. "
                            f"Open gt_overlay/*.jpg (red = `{best_h}`) and check how labels are converted.")
        elif normal > 1.5 * chance:
            verdicts.append(f"**A. GT placement looks right.** Normal placement contrast {normal} vs chance {chance} "
                            f"(best hypothesis: `{best_h}`). Still glance at gt_overlay/*.jpg to confirm.")
        else:
            verdicts.append(f"**A. Inconclusive.** Contrast under the normal placement ({normal}) is not clearly above chance ({chance}). "
                            "Thin low-contrast cracks do this too. Judge by eye from gt_overlay/*.jpg.")

    # ---------------------------------------------------------------- B/C: model
    backend = make_backend(args)
    configs = []
    results = {}
    inst_rows = []
    if backend is not None:
        configs = getattr(backend, "configs", None) or [(s, m) for s in args.sizes for m in args.modes]
        for (size, mode) in configs:
            name = f"{size}_{mode}"
            (out / "maps" / name).mkdir(parents=True, exist_ok=True)
            print(f"running {name} ...", flush=True)
            per_t = {t: [] for t in thresholds}
            max_in, max_out, vis = [], [], 0
            for s in samples:
                img = s["img"]
                H, W = img.shape[:2]
                gts = [raster(to_pixels(i["pts"], W, H, "normal"), W, H) for i in s["insts"]]
                prob = backend(img, size, mode, s["stem"])
                if prob.shape != (H, W):
                    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
                prob = np.nan_to_num(prob, nan=0.0)
                if gts:
                    union = np.any(gts, axis=0).astype(np.uint8)
                    far = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) == 0
                    max_out.append(float(prob[far].max()) if far.any() else 0.0)
                    m = instance_metrics(prob, gts, thresholds)
                    for ti, rows in m.items():
                        per_t[ti] += rows
                    for j, g in enumerate(gts):
                        mi = float(prob[g].max()) if g.any() else 0.0
                        max_in.append(mi)
                        row = {"config": name, "image": s["stem"], "inst": j, "gt_area_px": int(g.sum()),
                               "max_prob_in_gt": round(mi, 4)}
                        for ti in thresholds:
                            r = m[ti][j]
                            row[f"hit_t{ti}"] = int(r["hit_dil"] or r["best_iou"] >= 0.3)
                            row[f"iou_t{ti}"] = round(r["best_iou"], 3)
                        inst_rows.append(row)
                if vis < args.max_vis:
                    txt = f"{name} maxP_in_GT={max(max_in[-len(gts):]) if gts else 0:.2f}"
                    cv2.imwrite(str(out / "maps" / name / f"{s['stem']}.jpg"),
                                np.hstack([draw_gt(img, gts), heat_overlay(img, prob, gts, txt)]))
                    vis += 1
            summ = {"n_instances": len(max_in),
                    "median_max_prob_in_gt": round(float(np.median(max_in)), 4) if max_in else None,
                    "median_image_max_prob_outside_gt": round(float(np.median(max_out)), 4) if max_out else None,
                    "frac_instances_with_max_prob_ge": {str(v): round(float(np.mean(np.array(max_in) >= v)), 3)
                                                        for v in (0.05, 0.1, 0.25, 0.5, 0.9)} if max_in else {},
                    "by_threshold": {}}
            for t in thresholds:
                rows = per_t[t]
                n = max(1, len(rows))
                summ["by_threshold"][str(t)] = {
                    "recall_rule": round(sum(r["hit_dil"] or r["best_iou"] >= 0.3 for r in rows) / n, 3),
                    "recall_iou0.3": round(sum(r["best_iou"] >= 0.3 for r in rows) / n, 3),
                    "recall_iou0.5": round(sum(r["best_iou"] >= 0.5 for r in rows) / n, 3),
                    "pixel_recall": round(float(np.mean([r["pix_recall"] for r in rows])) if rows else 0.0, 3),
                }
            results[name] = summ

    # ---------------------------------------------------------------- verdicts for B / C
    if results:
        op = str(args.op_threshold)
        lo = str(min(thresholds))
        rec_op = {k: v["by_threshold"][op]["recall_rule"] for k, v in results.items()}
        rec_lo = {k: v["by_threshold"][lo]["recall_rule"] for k, v in results.items()}
        best_cfg = max(results, key=lambda k: (rec_op[k], rec_lo[k]))
        if all(v == 0 for v in rec_op.values()):
            if max(rec_lo.values()) >= 0.3:
                verdicts.append(f"**C. Threshold/calibration issue.** Recall is 0 at the operating threshold {args.op_threshold} in every "
                                f"configuration, but reaches {pct_str(max(rec_lo.values()))} at threshold {lo} "
                                f"(`{max(rec_lo, key=rec_lo.get)}`). The model responds on the cracks; its scores are too low or the "
                                "threshold was tuned on a different domain. Calibrate or use per-domain thresholds.")
            else:
                inn = max(v["median_max_prob_in_gt"] or 0 for v in results.values())
                outn = min(v["median_image_max_prob_outside_gt"] or 0 for v in results.values())
                verdicts.append(f"**B. The model barely responds on the GT cracks** (best median max-probability inside GT = {inn:.2f}; "
                                f"recall at {lo} is at most {pct_str(max(rec_lo.values()))}). That is a real domain/scale gap "
                                "unless check A flagged the labels. Next: look at maps/*/*.jpg; try tiling/upscaling and training with "
                                "crack-like content in this style.")
        else:
            verdicts.append(f"**Best configuration at threshold {args.op_threshold}:** `{best_cfg}` with rule-recall {pct_str(rec_op[best_cfg])} "
                            f"(at {lo}: {pct_str(rec_lo[best_cfg])}).")
        by_size = {}
        for k, v in rec_op.items():
            by_size.setdefault(k.split("_")[0], []).append(v)
        if len(by_size) > 1:
            m = {s: max(v) for s, v in by_size.items()}
            if max(m.values()) - min(m.values()) >= 0.2:
                verdicts.append("**Scale matters:** recall differs by input size " +
                                ", ".join(f"{s}px {pct_str(v)}" for s, v in sorted(m.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)) +
                                ". Inference size and crack width (tiling / scale augmentation) are worth fixing.")
        lb = [v for k, v in rec_op.items() if k.endswith("letterbox")]
        st = [v for k, v in rec_op.items() if k.endswith("stretch")]
        if lb and st and abs(max(lb) - max(st)) >= 0.2:
            verdicts.append(f"**Resize mode matters:** letterbox {pct_str(max(lb))} vs stretch {pct_str(max(st))}. "
                            "Roboflow-style stretched training data means a stretched input may suit the model better; "
                            "the retrain should use one consistent policy.")

    # ---------------------------------------------------------------- write outputs
    report = {"script_version": SCRIPT_VERSION, "data": str(args.data), "backend": args.backend,
              "sanity": sanity, "alignment": {"hypotheses": align, "chance_level": chance, "best": best_h},
              "thresholds": thresholds, "op_threshold": args.op_threshold, "configs": results, "verdicts": verdicts}
    (out / "results.json").write_text(json.dumps(report, indent=2))
    if inst_rows:
        with open(out / "instances.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(inst_rows[0].keys()))
            w.writeheader()
            w.writerows(inst_rows)

    L = ["# NCUGD pipeline check", "", f"Data: `{args.data}`  |  backend: `{args.backend}`", "",
         "## Verdicts", ""]
    L += [f"- {v}" for v in verdicts] or ["- (run with a model backend for checks B and C)"]
    L += ["", "## Data sanity", "", "```", json.dumps(sanity, indent=2), "```", "",
          "## A. GT alignment (mean local contrast under each coordinate hypothesis)", "",
          f"Chance level (normal polygons randomly shifted): **{chance}**", "",
          "| hypothesis | mean contrast | n |", "|---|---|---|"]
    for h, v in sorted(align.items(), key=lambda kv: -kv[1]["mean"]):
        L.append(f"| {h}{' (loader default)' if h == 'normal' else ''} | {v['mean']} | {v['n']} |")
    if results:
        L += ["", f"## B/C. Instance recall by configuration (rule: IoU>=0.3 or overlap after 3 px dilation), op threshold = {args.op_threshold}", ""]
        hdr = "| config | " + " | ".join(f"t={t}" for t in thresholds) + " | median maxP in GT | median maxP elsewhere | IoU>=0.5 @op |"
        L += [hdr, "|" + "---|" * (len(thresholds) + 4)]
        for k, v in results.items():
            cells = " | ".join(pct_str(v["by_threshold"][str(t)]["recall_rule"]) for t in thresholds)
            L.append(f"| {k} | {cells} | {v['median_max_prob_in_gt']} | {v['median_image_max_prob_outside_gt']} | "
                     f"{pct_str(v['by_threshold'][str(args.op_threshold)]['recall_iou0.5'])} |")
        L += ["", "Reading the table: if recall is high at low thresholds but 0 at the operating threshold, it is calibration. "
              "If 'median maxP in GT' is not above 'median maxP elsewhere', the model does not see these cracks at all."]
    L += ["", "## Files", "", "- `gt_overlay/` GT outlines (green; red = best alternative hypothesis if it differs)",
          "- `maps/<size>_<mode>/` left: image + GT, right: probability heat map + GT",
          "- `instances.csv` per-instance hits and max probability inside GT", ""]
    (out / "report.md").write_text("\n".join(L))

    print("\n".join(L[:40]))
    print(f"\nfull report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
