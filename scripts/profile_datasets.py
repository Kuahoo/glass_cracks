#!/usr/bin/env python3
"""profile_datasets.py - inspect crack/defect datasets BEFORE building any training mixture.

For each dataset folder it reports (and writes to profile/<dataset>.json):
  * image counts per split, file formats, image sizes, aspect ratios (stretch suspicion)
  * label format actually found: YOLO box / YOLO polygon / 9-token quad / COCO json / PNG masks
  * class IDs vs data.yaml names, instances per class, polygon vertices
  * empty-label rate, orphan images, orphan labels
  * crack stroke width (2*area/perimeter, pixels at native size and at a 640 px scale)
  * malformed lines, coordinates outside [0,1], degenerate polygons
  * warnings that say what to fix or verify before use

Only needs: numpy, opencv-python, Pillow.  PyYAML is optional (class names).
Raw data is only read, never modified.

Examples
  python profile_datasets.py                           # every folder in the bundle's datasets/ and processed/
  python profile_datasets.py path/to/dataset_a path/to/dataset_b
  python profile_datasets.py --children glass_crack_dataset_bundle/datasets
  python profile_datasets.py --mask-sample 100 --out profile_run2
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("opencv-python is required: pip install opencv-python")
try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required: pip install Pillow")

SCRIPT_VERSION = "1.0"
HERE = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train": "train", "training": "train", "valid": "valid", "val": "valid",
               "validation": "valid", "test": "test", "testing": "test"}
MASK_DIR_HINTS = re.compile(r"(^|[_\-])(mask|masks|annotation|annotations|annot|gt|groundtruth|"
                            r"ground_truth|segmentation|seg|sgm|binary)($|[_\-])", re.I)
LABEL_DIR_HINTS = re.compile(r"label", re.I)
SKIP_DIRS = {"__MACOSX", ".git", ".ipynb_checkpoints", "__pycache__", ".venv", "node_modules"}
REF = 640  # reference scale for stroke width


# ----------------------------------------------------------------------------- helpers
def pct(a, qs=(10, 50, 90)):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": round(float(np.percentile(a, q)), 4) for q in qs}


def summarize(a, nd=4):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "min": round(float(a.min()), nd), "mean": round(float(a.mean()), nd),
            "max": round(float(a.max()), nd), **pct(a)}


def split_of(path: Path, root: Path) -> str:
    for part in reversed(path.relative_to(root).parts[:-1]):
        if part.lower() in SPLIT_NAMES:
            return SPLIT_NAMES[part.lower()]
    return "-"


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def poly_width(xy: np.ndarray) -> float | None:
    """Mean stroke width of a thin polygon: 2*A/P (exact for a long strip). xy in pixels (N,2)."""
    if len(xy) < 3:
        return None
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    per = float(np.sum(np.hypot(x - np.roll(x, -1), y - np.roll(y, -1))))
    if per <= 0 or area <= 0:
        return None
    return float(2.0 * area / per)


def poly_area(xy: np.ndarray) -> float:
    if len(xy) < 3:
        return 0.0
    x, y = xy[:, 0], xy[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ----------------------------------------------------------------------------- scanning
def scan_dataset(root: Path):
    files = {"img": [], "txt": [], "json": [], "yaml": [], "zip": [], "other": 0}
    total_bytes = 0
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in fns:
            p = Path(dp) / fn
            ext = p.suffix.lower()
            try:
                total_bytes += p.stat().st_size
            except OSError:
                pass
            if ext in IMG_EXT:
                files["img"].append(p)
            elif ext == ".txt":
                files["txt"].append(p)
            elif ext == ".json":
                files["json"].append(p)
            elif ext in (".yaml", ".yml"):
                files["yaml"].append(p)
            elif ext in (".zip", ".tar", ".gz", ".7z", ".rar", ".tgz"):
                files["zip"].append(p)
            else:
                files["other"] += 1
    files["bytes"] = total_bytes
    return files


def is_mask_path(p: Path, root: Path) -> bool:
    rel = p.relative_to(root)
    if p.suffix.lower() != ".png" and p.suffix.lower() not in (".bmp", ".tif", ".tiff"):
        return False
    for part in rel.parts[:-1]:
        if MASK_DIR_HINTS.search(part):
            return True
    return bool(re.search(r"(_mask|_gt|_label|-mask|_seg)$", p.stem, re.I))


def read_names(yaml_paths):
    names = {}
    for yp in yaml_paths[:5]:
        try:
            import yaml  # type: ignore
            d = yaml.safe_load(yp.read_text(errors="ignore"))
            n = d.get("names") if isinstance(d, dict) else None
            if isinstance(n, dict):
                names[str(yp)] = {int(k): v for k, v in n.items()}
            elif isinstance(n, list):
                names[str(yp)] = {i: v for i, v in enumerate(n)}
        except Exception:
            try:
                txt = yp.read_text(errors="ignore")
                m = re.search(r"names\s*:\s*\[(.*?)\]", txt, re.S)
                if m:
                    lst = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
                    names[str(yp)] = {i: v for i, v in enumerate(lst)}
            except Exception:
                pass
    return names


# ----------------------------------------------------------------------------- images
def profile_images(imgs, root, max_read):
    """imgs: list of Paths (non-mask). Returns (info dict, size lookup {(split, stem): (w,h)})."""
    by_split = collections.Counter()
    fmt = collections.Counter()
    modes = collections.Counter()
    sizes = collections.Counter()
    ws, hs, ars = [], [], []
    lookup = {}
    unreadable = []
    dup_keys = collections.Counter()
    for i, p in enumerate(imgs):
        sp = split_of(p, root)
        by_split[sp] += 1
        fmt[p.suffix.lower()] += 1
        key = (sp, p.stem)
        dup_keys[key] += 1
        if i >= max_read and key in lookup:
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
                if i < max_read:
                    modes[im.mode] += 1
        except Exception:
            unreadable.append(str(p.relative_to(root)))
            continue
        lookup.setdefault(key, (w, h))
        if i < max_read:
            sizes[(w, h)] += 1
            ws.append(w)
            hs.append(h)
            ars.append(w / h if h else 0)
    n_read = len(ws)
    info = {
        "count": len(imgs),
        "per_split": dict(by_split),
        "formats": dict(fmt),
        "modes_sampled": dict(modes),
        "n_size_read": n_read,
        "top_sizes": [{"wh": list(k), "n": v} for k, v in sizes.most_common(6)],
        "n_distinct_sizes": len(sizes),
        "width": summarize(ws, 1), "height": summarize(hs, 1),
        "aspect_w_over_h": summarize(ars, 3),
        "unreadable": unreadable[:20], "n_unreadable": len(unreadable),
        "stem_collisions_within_split": int(sum(1 for v in dup_keys.values() if v > 1)),
    }
    if n_read:
        sq = sum(1 for w, h in zip(ws, hs) if w == h) / n_read
        info["fraction_square"] = round(sq, 3)
    return info, lookup


# ----------------------------------------------------------------------------- YOLO labels
def classify_line(tokens):
    """Return (kind, cls, coords ndarray or None)."""
    try:
        vals = [float(t) for t in tokens]
    except ValueError:
        return "non_numeric", None, None
    n = len(vals)
    cls = vals[0]
    if n == 5:
        return "box", cls, np.array(vals[1:])
    if n == 9:
        return "quad9", cls, np.array(vals[1:]).reshape(4, 2)
    if n >= 7 and n % 2 == 1:
        return "polygon", cls, np.array(vals[1:]).reshape(-1, 2)
    if n == 6 or (n > 5 and n % 2 == 0):
        return "bad_token_count", cls, None
    return "bad_token_count", cls, None


def profile_label_dir(files, root, size_lookup, names_by_yaml, max_lines_sample=3):
    """files: list of txt Paths that live in one 'labels*' directory (same parent)."""
    kinds = collections.Counter()
    per_cls = collections.defaultdict(lambda: {"instances": 0, "images": set(), "verts": [], "rel_area": [],
                                                 "width_native": [], "width_ref": [], "box_w_ref": [], "box_h_ref": []})
    per_img_counts = []
    empty_files = 0
    non_numeric_samples, bad_samples = [], []
    out_of_range = 0
    degenerate = 0
    no_size = 0
    pixel_units = 0  # lines with coords > 1.5 -> probably not normalized
    for f in files:
        sp = split_of(f, root)
        try:
            lines = [ln.split() for ln in f.read_text(errors="ignore").splitlines() if ln.strip()]
        except Exception:
            continue
        if not lines:
            empty_files += 1
            per_img_counts.append(0)
            continue
        wh = size_lookup.get((sp, f.stem))
        if wh is None:
            no_size += 1
        n_inst = 0
        for tokens in lines:
            kind, cls, xy = classify_line(tokens)
            kinds[kind] += 1
            if kind == "non_numeric":
                if len(non_numeric_samples) < max_lines_sample:
                    non_numeric_samples.append(" ".join(tokens)[:120])
                continue
            if kind == "bad_token_count":
                if len(bad_samples) < max_lines_sample:
                    bad_samples.append(" ".join(tokens)[:120])
                continue
            n_inst += 1
            c = int(cls)
            d = per_cls[c]
            d["instances"] += 1
            d["images"].add((sp, f.stem))
            if xy is not None:
                if np.any(xy > 1.5):
                    pixel_units += 1
                if np.any(xy < -1e-6) or np.any(xy > 1 + 1e-6):
                    out_of_range += 1
            if kind == "quad9":
                # 9 tokens = class + 4 points: oriented box (DOTA-like) OR a 4-point polygon. Report size only.
                d["verts"].append(4)
                if wh is not None:
                    W, H = wh
                    (_, _), (bw, bh), _ = cv2.minAreaRect((xy * np.array([W, H])).astype(np.float32))
                    d["box_w_ref"].append(max(bw, bh) * REF / max(W, H))
                    d["box_h_ref"].append(min(bw, bh) * REF / max(W, H))
                    d["rel_area"].append(poly_area(xy * np.array([W, H])) / (W * H))
            elif kind == "polygon":
                d["verts"].append(len(xy))
                if len(xy) < 3:
                    degenerate += 1
                if wh is not None:
                    W, H = wh
                    pts = xy * np.array([W, H])
                    a = poly_area(pts)
                    d["rel_area"].append(a / (W * H))
                    if a <= 0:
                        degenerate += 1
                    wd = poly_width(pts)
                    if wd is not None:
                        d["width_native"].append(wd)
                        d["width_ref"].append(wd * REF / max(W, H))
            elif kind == "box" and wh is not None:
                W, H = wh
                d["box_w_ref"].append(xy[2] * W * REF / max(W, H))
                d["box_h_ref"].append(xy[3] * H * REF / max(W, H))
                d["rel_area"].append(xy[2] * xy[3])
        per_img_counts.append(n_inst)
    classes = {}
    for c, d in sorted(per_cls.items()):
        entry = {"instances": d["instances"], "images_with_class": len(d["images"])}
        if d["verts"]:
            entry["vertices"] = summarize(d["verts"], 1)
        if d["rel_area"]:
            entry["rel_area"] = summarize(d["rel_area"], 5)
        if d["width_native"]:
            entry["stroke_width_native_px"] = summarize(d["width_native"], 2)
            entry["stroke_width_at_640_px"] = summarize(d["width_ref"], 2)
        if d["box_w_ref"]:
            entry["box_w_at_640_px"] = summarize(d["box_w_ref"], 1)
            entry["box_h_at_640_px"] = summarize(d["box_h_ref"], 1)
        classes[str(c)] = entry
    n_files = len(files)
    return {
        "label_files": n_files,
        "line_kinds": dict(kinds),
        "empty_label_files": empty_files,
        "empty_label_rate": round(empty_files / n_files, 4) if n_files else None,
        "instances_per_image": summarize(per_img_counts, 2),
        "classes": classes,
        "out_of_range_lines": out_of_range,
        "pixel_unit_lines": pixel_units,
        "degenerate_polygons": degenerate,
        "files_without_matching_image_size": no_size,
        "non_numeric_samples": non_numeric_samples,
        "bad_token_samples": bad_samples,
    }


# ----------------------------------------------------------------------------- COCO json
def profile_coco(jpaths, root, size_cap_mb=400):
    out = []
    for jp in jpaths:
        try:
            if jp.stat().st_size > size_cap_mb * 1024 * 1024:
                out.append({"file": str(jp.relative_to(root)), "skipped": "too large"})
                continue
            d = json.loads(jp.read_text(errors="ignore"))
        except Exception as e:
            out.append({"file": str(jp.relative_to(root)), "skipped": f"not parseable: {e}"[:100]})
            continue
        if not (isinstance(d, dict) and "images" in d and "annotations" in d):
            continue
        cats = {c.get("id"): c.get("name") for c in d.get("categories", [])}
        imgs = {im["id"]: im for im in d["images"]}
        per_cat = collections.defaultdict(lambda: {"instances": 0, "polygons": 0, "rle": 0, "bbox_only": 0,
                                                   "width_ref": [], "area_rel": []})
        for a in d["annotations"]:
            pc = per_cat[a.get("category_id")]
            pc["instances"] += 1
            seg = a.get("segmentation")
            im = imgs.get(a.get("image_id"))
            if isinstance(seg, list) and seg:
                pc["polygons"] += 1
                if im:
                    W, H = im.get("width"), im.get("height")
                    for poly in seg:
                        arr = np.asarray(poly, dtype=float).reshape(-1, 2)
                        wd = poly_width(arr)
                        if wd is not None and W and H:
                            pc["width_ref"].append(wd * REF / max(W, H))
                    if W and H and a.get("area"):
                        pc["area_rel"].append(a["area"] / (W * H))
            elif isinstance(seg, dict):
                pc["rle"] += 1
            else:
                pc["bbox_only"] += 1
        covered = {a.get("image_id") for a in d["annotations"]}
        out.append({
            "file": str(jp.relative_to(root)),
            "images": len(d["images"]), "annotations": len(d["annotations"]),
            "images_without_annotations": len([i for i in imgs if i not in covered]),
            "categories": {str(k): v for k, v in cats.items()},
            "per_category": {str(cats.get(k, k)): {
                "instances": v["instances"], "polygons": v["polygons"], "rle": v["rle"],
                "bbox_only": v["bbox_only"],
                "stroke_width_at_640_px": summarize(v["width_ref"], 2),
                "rel_area": summarize(v["area_rel"], 5)} for k, v in per_cat.items()},
        })
    return out


# ----------------------------------------------------------------------------- masks
def profile_masks(mask_paths, root, n_sample, seed=0):
    rng = random.Random(seed)
    sample = mask_paths if len(mask_paths) <= n_sample else rng.sample(mask_paths, n_sample)
    uniq_sets = collections.Counter()
    pos_frac, comp_counts, widths_native, widths_ref = [], [], [], []
    channels = collections.Counter()
    dtypes = collections.Counter()
    empty = 0
    unreadable = 0
    for mp in sample:
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if m is None:
            unreadable += 1
            continue
        dtypes[str(m.dtype)] += 1
        channels[m.shape[2] if m.ndim == 3 else 1] += 1
        if m.ndim == 3:
            m = m[..., 0] if m.shape[2] != 4 else m[..., 3]
        vals = np.unique(m)
        uniq_sets[tuple(int(v) for v in vals[:6]) + (("...",) if len(vals) > 6 else ())] += 1
        fg = (m > 0).astype(np.uint8)
        frac = float(fg.mean())
        pos_frac.append(frac)
        if frac == 0:
            empty += 1
            continue
        H, W = fg.shape
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        comp_counts.append(n - 1)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            a = cv2.contourArea(c)
            p = cv2.arcLength(c, True)
            if p > 0 and a >= 3:
                w = 2.0 * a / p
                widths_native.append(w)
                widths_ref.append(w * REF / max(W, H))
    return {
        "n_masks_total": len(mask_paths), "n_sampled": len(sample), "unreadable": unreadable,
        "value_sets_top": [{"values": list(k), "n": v} for k, v in uniq_sets.most_common(5)],
        "channels": dict(channels), "dtypes": dict(dtypes),
        "empty_masks": empty, "empty_rate": round(empty / max(1, len(sample) - unreadable), 4),
        "foreground_fraction": summarize(pos_frac, 5),
        "components_per_mask": summarize(comp_counts, 1),
        "stroke_width_native_px": summarize(widths_native, 2),
        "stroke_width_at_640_px": summarize(widths_ref, 2),
        "note": "width = 2*area/perimeter per connected blob: a lower bound for blobs that touch/branch.",
    }


# ----------------------------------------------------------------------------- per-dataset
def profile_dataset(root: Path, args) -> dict:
    t0 = time.time()
    files = scan_dataset(root)
    mask_files = [p for p in files["img"] if is_mask_path(p, root)]
    mask_set = set(mask_files)
    image_files = [p for p in files["img"] if p not in mask_set]
    rep: dict = {"dataset": root.name, "path": str(root), "script_version": SCRIPT_VERSION,
                 "size_on_disk": human(files["bytes"]), "bytes": files["bytes"],
                 "archives_found": [str(p.relative_to(root)) for p in files["zip"][:10]],
                 "n_archives": len(files["zip"])}
    warnings = []

    img_info, size_lookup = profile_images(image_files, root, args.max_image_read)
    rep["images"] = img_info

    names = read_names(files["yaml"])
    if names:
        rep["class_names_from_yaml"] = {str(Path(k).relative_to(root)): v for k, v in names.items()}
    # group txt files by parent dir, keep only dirs that look like label dirs or contain YOLO-like lines
    by_dir = collections.defaultdict(list)
    for t in files["txt"]:
        if t.name.lower() in ("readme.txt", "requirements.txt", "license.txt", "classes.txt", "labels.txt"):
            continue
        by_dir[t.parent].append(t)
    label_dirs = {}
    image_dirs = {p.parent for p in image_files}
    for d, fl in by_dir.items():
        # label dir if name hints OR many txt files whose stems match image stems somewhere
        stems = {(split_of(f, root), f.stem) for f in fl[:200]}
        match = len(stems & set(size_lookup.keys()))
        if LABEL_DIR_HINTS.search(d.name) or match >= max(1, 0.5 * len(stems)):
            label_dirs[d] = fl
    rep["label_dirs"] = {}
    all_img_keys = set(size_lookup.keys())
    for d, fl in sorted(label_dirs.items(), key=lambda kv: str(kv[0])):
        key = str(d.relative_to(root)) or "."
        # profile per split inside this dir tree is already encoded by split_of; group by parent dir only
        r = profile_label_dir(fl, root, size_lookup, names)
        lab_keys = {(split_of(f, root), f.stem) for f in fl}
        lab_splits = {k[0] for k in lab_keys}
        scope = all_img_keys if "-" in lab_splits else {k for k in all_img_keys if k[0] in lab_splits}
        r["images_with_no_label_file"] = len(scope - lab_keys)
        r["label_files_without_image"] = len(lab_keys - all_img_keys)
        rep["label_dirs"][key] = r
    # when many sibling split dirs exist (train/labels, valid/labels, ...) fold them for readability
    rep["label_dirs_merged"] = merge_label_dirs(rep["label_dirs"])

    coco = profile_coco(files["json"], root)
    if coco:
        rep["coco_json"] = coco
    if mask_files:
        rep["masks"] = profile_masks(mask_files, root, args.mask_sample)

    # ---- label type summary + warnings
    kinds = collections.Counter()
    for r in rep["label_dirs"].values():
        kinds.update(r["line_kinds"])
    label_types = []
    if kinds.get("polygon"):
        label_types.append("YOLO polygon")
    if kinds.get("box"):
        label_types.append("YOLO box")
    if kinds.get("quad9"):
        label_types.append("quad/OBB (9 tokens)")
        warnings.append("Lines with 9 tokens found: either oriented boxes (4 corner points) or 4-point polygons. "
                        "Open a few files and confirm before converting.")
    if coco:
        label_types.append("COCO json")
    if mask_files:
        label_types.append("PNG masks")
    rep["label_types"] = label_types or ["none found"]

    n_img = img_info["count"]
    if n_img == 0:
        warnings.append("No image files found.")
    if not label_types:
        warnings.append("No labels recognized (no YOLO txt, COCO json or mask PNGs).")
    if "YOLO box" in label_types and "YOLO polygon" not in label_types and not mask_files and not any(
            c.get("polygons") for cj in coco for c in cj.get("per_category", {}).values()):
        warnings.append("Boxes only: no masks. Cannot be used for mask IoU / segmentation training as-is.")
    if img_info.get("fraction_square", 0) >= 0.95 and img_info["n_distinct_sizes"] <= 2 and n_img > 20:
        warnings.append("All images share one square size: likely resized/stretched (Roboflow-style). "
                        "Crack geometry/aspect may be distorted; physical width is not recoverable.")
    if img_info["n_unreadable"]:
        warnings.append(f"{img_info['n_unreadable']} unreadable image(s): {img_info['unreadable'][:3]}")
    if rep["n_archives"]:
        warnings.append(f"{rep['n_archives']} archive file(s) in the folder (possibly unextracted or leftover).")
    if img_info["stem_collisions_within_split"]:
        warnings.append(f"{img_info['stem_collisions_within_split']} image stem(s) occur more than once within a split "
                        "(e.g. forward/backward pairs); label matching by stem is ambiguous.")
    for key, r in rep["label_dirs"].items():
        tag = f"labels dir '{key}'"
        if r["line_kinds"].get("non_numeric"):
            warnings.append(f"{tag}: {r['line_kinds']['non_numeric']} non-numeric line(s), e.g. {r['non_numeric_samples'][:1]}. "
                            "Probably not YOLO format (DOTA / VOC-style?).")
        if r["line_kinds"].get("bad_token_count"):
            warnings.append(f"{tag}: {r['line_kinds']['bad_token_count']} line(s) with a bad token count, e.g. {r['bad_token_samples'][:1]}")
        if r["pixel_unit_lines"]:
            warnings.append(f"{tag}: {r['pixel_unit_lines']} line(s) have coordinates > 1.5: labels look like pixel units, not normalized.")
        elif r["out_of_range_lines"]:
            warnings.append(f"{tag}: {r['out_of_range_lines']} line(s) have coordinates outside [0,1]; clip them.")
        if r["degenerate_polygons"]:
            warnings.append(f"{tag}: {r['degenerate_polygons']} degenerate polygon(s) (<3 points or zero area).")
        if r["empty_label_rate"] is not None and r["empty_label_rate"] > 0.5:
            warnings.append(f"{tag}: {r['empty_label_rate']:.0%} of label files are empty.")
        if r["images_with_no_label_file"] > 0 and r["label_files"] > 0:
            warnings.append(f"{tag}: {r['images_with_no_label_file']} image(s) have no label file there "
                            "(treat as unlabeled, not as negatives, unless confirmed).")
        if r["label_files_without_image"] > 0:
            warnings.append(f"{tag}: {r['label_files_without_image']} label file(s) have no matching image.")
        # class ids vs yaml
        if names:
            known = set()
            for nm in names.values():
                known |= set(nm.keys())
            bad = [c for c in r["classes"] if int(c) not in known]
            if bad:
                warnings.append(f"{tag}: class id(s) {bad} not in data.yaml names {sorted(known)}.")
    if mask_files:
        ms = rep["masks"]
        if ms["empty_rate"] > 0.3:
            warnings.append(f"Masks: {ms['empty_rate']:.0%} of sampled masks are empty.")
        w = ms["stroke_width_at_640_px"]
        if w.get("n") and w.get("p50") is not None and w["p50"] > 6:
            warnings.append(f"Masks: median stroke width ~{w['p50']} px at 640 scale: thick strokes; "
                            "use for localization pre-training only (see plan Section 5).")
        if len(image_files) and len(mask_files) and abs(len(image_files) - len(mask_files)) > 0.05 * len(image_files):
            warnings.append(f"{len(image_files)} images but {len(mask_files)} masks: counts differ by >5%.")
    # CUBIT-style scale flag
    if img_info["width"].get("max") and img_info["width"]["max"] > 2000:
        warnings.append("Large images (>2000 px wide): tile before training at 640; cracks become invisible if downscaled whole.")
    rep["warnings"] = warnings
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def merge_label_dirs(label_dirs: dict) -> dict:
    """Fold sibling label dirs with the same basename (train/labels, valid/labels...) into one summary."""
    groups = collections.defaultdict(list)
    for k, v in label_dirs.items():
        groups[Path(k).name].append(v)
    out = {}
    for name, lst in groups.items():
        kinds = collections.Counter()
        cls = collections.defaultdict(lambda: {"instances": 0, "images_with_class": 0})
        n_files = empty = 0
        for r in lst:
            kinds.update(r["line_kinds"])
            n_files += r["label_files"]
            empty += r["empty_label_files"]
            for c, d in r["classes"].items():
                cls[c]["instances"] += d["instances"]
                cls[c]["images_with_class"] += d["images_with_class"]
        out[name] = {"label_files": n_files, "empty_label_files": empty, "line_kinds": dict(kinds),
                     "instances_by_class": {c: d["instances"] for c, d in sorted(cls.items())},
                     "images_with_class": {c: d["images_with_class"] for c, d in sorted(cls.items())},
                     "n_dirs": len(lst)}
    return out


# ----------------------------------------------------------------------------- output
def one_line(rep: dict) -> dict:
    imgs = rep["images"]
    top = imgs["top_sizes"][0] if imgs["top_sizes"] else None
    width_vals = []
    for r in rep["label_dirs"].values():
        for c in r["classes"].values():
            w = c.get("stroke_width_at_640_px")
            if w and w.get("n"):
                width_vals.append((w["n"], w["p50"]))
    if "masks" in rep:
        w = rep["masks"]["stroke_width_at_640_px"]
        if w.get("n"):
            width_vals.append((w["n"], w["p50"]))
    med = None
    if width_vals:
        tot = sum(n for n, _ in width_vals)
        med = round(sum(n * m for n, m in width_vals) / tot, 2)
    empties = [r["empty_label_rate"] for r in rep["label_dirs"].values() if r["empty_label_rate"] is not None]
    return {
        "dataset": rep["dataset"], "images": imgs["count"], "size_on_disk": rep["size_on_disk"],
        "top_size": f"{top['wh'][0]}x{top['wh'][1]}" if top else "",
        "n_sizes": imgs["n_distinct_sizes"],
        "label_types": "; ".join(rep["label_types"]),
        "empty_label_rate": round(float(np.mean(empties)), 3) if empties else "",
        "stroke_width_p50_at_640": med if med is not None else "",
        "n_warnings": len(rep["warnings"]),
    }


def write_outputs(reps, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in reps:
        (out_dir / f"{r['dataset']}.json").write_text(json.dumps(r, indent=2, default=str))
    rows = [one_line(r) for r in reps]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["dataset"])
        w.writeheader()
        w.writerows(rows)
    lines = ["# Dataset profile summary", "",
             "| dataset | images | disk | most common size (#sizes) | label type | empty-label rate | stroke width p50 @640 | warnings |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['images']} | {row['size_on_disk']} | {row['top_size']} ({row['n_sizes']}) | "
                     f"{row['label_types']} | {row['empty_label_rate']} | {row['stroke_width_p50_at_640']} | {row['n_warnings']} |")
    lines += ["", "## Warnings per dataset", ""]
    for r in reps:
        lines.append(f"### {r['dataset']}")
        if r["warnings"]:
            lines += [f"- {w}" for w in r["warnings"]]
        else:
            lines.append("- none")
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))


def print_report(rep):
    print("=" * 78)
    print(f"{rep['dataset']}   [{rep['size_on_disk']}, {rep['seconds']} s]")
    im = rep["images"]
    print(f"  images: {im['count']}  splits: {im['per_split']}  formats: {im['formats']}")
    if im["top_sizes"]:
        print("  sizes:  " + ", ".join(f"{s['wh'][0]}x{s['wh'][1]} (n={s['n']})" for s in im["top_sizes"][:4])
              + (f"  [+{im['n_distinct_sizes'] - 4} more]" if im["n_distinct_sizes"] > 4 else ""))
    print(f"  labels: {', '.join(rep['label_types'])}")
    for name, r in rep["label_dirs_merged"].items():
        print(f"    '{name}': files={r['label_files']} empty={r['empty_label_files']} lines={r['line_kinds']} "
              f"instances/class={r['instances_by_class']}")
    for key, r in rep["label_dirs"].items():
        for c, d in r["classes"].items():
            w = d.get("stroke_width_at_640_px")
            if w and w.get("n"):
                print(f"    {key} class {c}: width@640 p10/p50/p90 = {w['p10']}/{w['p50']}/{w['p90']} px "
                      f"(n={w['n']}), vertices mean {d['vertices']['mean'] if 'vertices' in d else '-'}")
                break
    for cj in rep.get("coco_json", []):
        if "skipped" in cj:
            print(f"    COCO {cj['file']}: skipped ({cj['skipped']})")
        else:
            print(f"    COCO {cj['file']}: {cj['images']} images, {cj['annotations']} annotations, "
                  f"cats={list(cj['categories'].values())}")
    if "masks" in rep:
        m = rep["masks"]
        w = m["stroke_width_at_640_px"]
        print(f"  masks: {m['n_masks_total']} files (sampled {m['n_sampled']}), values {m['value_sets_top'][:2]}, "
              f"empty {m['empty_rate']:.0%}, width@640 p50 = {w.get('p50')}")
    if rep["warnings"]:
        print("  WARNINGS:")
        for w in rep["warnings"]:
            print(f"   - {w}")
    else:
        print("  warnings: none")


def default_targets():
    bundle = HERE / "glass_crack_dataset_bundle"
    out = []
    for sub in ("datasets", "processed"):
        d = bundle / sub
        if d.is_dir():
            out += sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", type=Path, help="dataset folders (each is profiled as one dataset)")
    ap.add_argument("--children", action="store_true",
                    help="treat each immediate sub-folder of every target as its own dataset")
    ap.add_argument("--out", type=Path, default=HERE / "profile", help="output folder (default: ./profile next to script)")
    ap.add_argument("--mask-sample", type=int, default=300, help="max mask PNGs to read per dataset (default 300)")
    ap.add_argument("--max-image-read", type=int, default=20000,
                    help="max images whose size is read for the size statistics (default 20000)")
    ap.add_argument("--quiet", action="store_true", help="only write files, print the one-line table")
    args = ap.parse_args(argv)

    targets = list(args.targets)
    if not targets:
        targets = default_targets()
        if not targets:
            ap.error("no targets given and no glass_crack_dataset_bundle/ found next to the script")
    elif args.children:
        expanded = []
        for t in targets:
            expanded += sorted(p for p in t.iterdir() if p.is_dir() and not p.name.startswith("."))
        targets = expanded
    targets = [t.resolve() for t in targets]
    missing = [t for t in targets if not t.is_dir()]
    if missing:
        ap.error(f"not a directory: {missing[0]}")

    reps = []
    for t in targets:
        print(f"profiling {t} ...", flush=True)
        rep = profile_dataset(t, args)
        reps.append(rep)
        if not args.quiet:
            print_report(rep)
    write_outputs(reps, args.out)
    print("\n" + "-" * 78)
    print(f"wrote {len(reps)} profile(s) + summary.md / summary.csv to {args.out}")
    for r in reps:
        row = one_line(r)
        print(f"  {row['dataset']:<28} {row['images']:>6} imgs  {row['label_types']:<28} warnings: {row['n_warnings']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
