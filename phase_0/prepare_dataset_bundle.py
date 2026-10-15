#!/usr/bin/env python3
"""phase_0/prepare_dataset_bundle.py

Full multi-dataset preprocessor and harmonizer for Glass Crack Segmentation.
Every positive source is mapped to a single YOLO-seg class ``0: crack``
(cracks, breaks and scratches), every negative source contributes images or
tiles with *no* defect of any kind.

Sources:
  - SSGD (COCO, flat rectangle "segmentations" = box-quality masks;
    crack / broken / scratch / broken-membrane)
  - CarDD_release (COCO polygons; crack & glass shatter)
  - agdd_aircraft_glass (oriented-box labels in data/labels/<split>/;
    scratches + crack, both image views).  Excluded by default because the
    model implementation plan reserves AGDD as an external test set; pass
    ``--exclude`` with no arguments (or a list without agdd) to train on it.
  - crackSegUltralytics (YOLO-seg, subsampled)
  - crackseg9k (Parquet with base64 PNG image/mask, subsampled)
  - glass_defect_detection2 (YOLO boxes; crack + scratch, bubble/inclusion dropped)
  - Clean negatives: defect-free CUBIT tiles, empty-label glass_segmentation_2026
    images, empty-mask fcd_108_facade images
  - Isolated external test benchmark: ncugd_glass_crack (never tiled or split)

Train/val assignment is deterministic and group-aware: every tile, view and
Roboflow augmentation of the same source photo lands in the same split.

Usage:
  python phase_0/prepare_dataset_bundle.py --data-root data --out-dir data/processed_bundle_v1
  python phase_0/prepare_dataset_bundle.py --limit 20 --out-dir /tmp/bundle_smoke   # quick smoke run
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXTERNAL_SPLIT = "ncugd_test_external"
BUNDLE_ENTRIES = {"images", "labels", "staged", "data.yaml", "summary_report.json", "manifest.csv"}
DATASET_KEYS = ("gdd2", "ssgd", "agdd", "cardd", "ucrack", "cs9k", "neg_cubit", "neg_gseg", "neg_fcd", "ncugd")

Label = Tuple[int, List[float]]


# =====================================================================
# 1. Geometry & Polygon Utilities
# =====================================================================
def sanitize_polygon(coords: Sequence[float]) -> Optional[List[float]]:
    """Clips normalized polygon coords to [0, 1]; rejects malformed or point-like polygons."""
    pts = np.asarray(coords, dtype=np.float64)
    if pts.size < 6 or pts.size % 2 or not np.isfinite(pts).all():
        return None
    pts = np.clip(pts.reshape(-1, 2), 0.0, 1.0)
    if np.ptp(pts[:, 0]) == 0 and np.ptp(pts[:, 1]) == 0:
        return None
    return pts.ravel().round(6).tolist()


def pixel_polygon_to_norm(coords: Sequence[float], img_w: int, img_h: int) -> Optional[List[float]]:
    pts = np.asarray(coords, dtype=np.float64)
    if pts.size < 6 or pts.size % 2:
        return None
    return sanitize_polygon((pts.reshape(-1, 2) / [img_w, img_h]).ravel())


def box_to_polygon(cx: float, cy: float, w: float, h: float) -> List[float]:
    """Converts normalized YOLO box (cx, cy, w, h) to 4-corner polygon rectangle."""
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _simplify_contour(cnt: np.ndarray, epsilon: float) -> np.ndarray:
    approx = cv2.approxPolyDP(cnt, epsilon, closed=True).reshape(-1, 2)
    if len(approx) < 3:
        # 1-px wide straight cracks collapse to a segment; keep them as a thin rotated rectangle.
        approx = cv2.boxPoints(cv2.minAreaRect(cnt))
    return approx.astype(np.float64)


def _splice_holes(outer: np.ndarray, holes: List[np.ndarray]) -> np.ndarray:
    """Joins hole rings into the outer ring with zero-width bridges so fillPoly leaves them empty."""
    poly = outer
    for hole in holes:
        d = ((poly[:, None, :] - hole[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(int(d.argmin()), d.shape)
        ring = np.roll(hole, -j, axis=0)
        poly = np.concatenate([poly[: i + 1], ring, ring[:1], poly[i:]])
    return poly


def mask_to_yolo_polygons(
    mask: np.ndarray,
    min_pixels: int = 12,
    epsilon: float = 1.0,
    min_hole_pixels: float = 4.0,
) -> List[List[float]]:
    """Converts a binary mask (H, W) to normalized YOLOv8 polygons (one per connected component).

    Components are filtered by pixel count, not contour area, so thin cracks survive.
    Holes (closed crack loops) are preserved instead of being filled in.
    """
    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    polygons = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < min_pixels:
            continue
        comp = (cc[y : y + bh, x : x + bw] == i).astype(np.uint8)
        contours, hierarchy = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]
        for k, cnt in enumerate(contours):
            if hierarchy[k][3] >= 0:
                continue
            ring = _simplify_contour(cnt, epsilon)
            holes = [
                _simplify_contour(contours[c], epsilon)
                for c in range(len(contours))
                if hierarchy[c][3] == k and cv2.contourArea(contours[c]) >= min_hole_pixels
            ]
            if holes:
                ring = _splice_holes(ring, holes)
            poly = sanitize_polygon(((ring + [x, y]) / [w, h]).ravel())
            if poly:
                polygons.append(poly)
    return polygons


def clip_polygon_to_tile(
    pts_px: np.ndarray, tx: int, ty: int, tw: int, th: int, min_pixels: int = 8
) -> List[List[float]]:
    """Clips a pixel-space polygon (N, 2) to a tile window and re-normalizes to the tile."""
    canvas = np.zeros((th, tw), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.round(pts_px - [tx, ty]).astype(np.int32)], 1)
    return mask_to_yolo_polygons(canvas, min_pixels=min_pixels, epsilon=0.8)


def get_tile_coordinates(img_w: int, img_h: int, tile_size: int, overlap_ratio: float = 0.20):
    def starts(length: int) -> List[int]:
        if length <= tile_size:
            return [0]
        step = max(1, int(tile_size * (1.0 - overlap_ratio)))
        s = list(range(0, length - tile_size + 1, step))
        if s[-1] + tile_size < length:
            s.append(length - tile_size)
        return s

    return [(x, y, min(tile_size, img_w - x), min(tile_size, img_h - y)) for y in starts(img_h) for x in starts(img_w)]


# =====================================================================
# 2. File / Label Helpers
# =====================================================================
_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)
_EXT_SUFFIX = re.compile(r"_(jpe?g|png|bmp|webp|tiff?)$", re.IGNORECASE)


def source_key(stem: str) -> str:
    """Strips Roboflow '.rf.<hash>' and '_jpg'-style suffixes so augmented copies share a key."""
    s = _RF_SUFFIX.sub("", stem)
    while True:
        t = _EXT_SUFFIX.sub("", s)
        if t == s:
            return s
        s = t


def list_images(root: Path) -> List[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMG_EXTENSIONS
        and "__MACOSX" not in p.parts
        and not p.name.startswith("._")
    )


def cap(items: List, n: Optional[int], rng: random.Random) -> List:
    if n is not None and len(items) > n:
        return rng.sample(items, n)
    return items


def yolo_label_path(img_p: Path) -> Path:
    """Ultralytics convention: .../images/<...>/x.jpg -> .../labels/<...>/x.txt"""
    parts = list(img_p.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img_p.with_suffix(".txt")


def parse_yolo_labels(lbl_p: Optional[Path], keep: Optional[Dict[int, int]]) -> List[Label]:
    """Reads YOLO box (5 tokens) or polygon (>= 7 tokens) lines.

    ``keep`` maps source class id -> output class id; other classes are dropped.
    ``keep=None`` keeps every class with its original id.
    """
    labels: List[Label] = []
    if lbl_p is None or not lbl_p.exists():
        return labels
    for line in lbl_p.read_text().splitlines():
        t = line.split()
        if not t:
            continue
        try:
            cls = int(float(t[0]))
            vals = [float(v) for v in t[1:]]
        except ValueError:
            continue
        if keep is not None and cls not in keep:
            continue
        if len(vals) == 4:
            poly = sanitize_polygon(box_to_polygon(*vals))
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            poly = sanitize_polygon(vals)
        else:
            continue
        if poly:
            labels.append((keep[cls] if keep is not None else cls, poly))
    return labels


def coco_polygons(ann: dict) -> List[List[float]]:
    """Returns pixel polygons from a COCO annotation (list-of-lists or flat list; RLE skipped)."""
    seg = ann.get("segmentation")
    if isinstance(seg, list) and seg:
        if all(isinstance(v, (int, float)) for v in seg):
            return [seg]
        return [s for s in seg if isinstance(s, list) and len(s) >= 6]
    if not seg and ann.get("bbox"):
        x, y, w, h = ann["bbox"]
        return [[x, y, x + w, y, x + w, y + h, x, y + h]]
    return []


# =====================================================================
# 3. Slicing & Storage Engine
# =====================================================================
class BundleWriter:
    def __init__(self, out_dir: Path, val_ratio: float, tile_size: Optional[int], overlap: float):
        self.out = out_dir
        self.val_ratio = val_ratio
        self.tile_size = tile_size
        self.overlap = overlap
        self.used_stems: set = set()
        self.manifest: List[dict] = []
        self.stats: Dict[str, Counter] = defaultdict(Counter)

    def split_for(self, group: str) -> str:
        h = int(hashlib.md5(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return "val" if h < self.val_ratio else "train"

    def _unique_stem(self, stem: str) -> str:
        out, n = stem, 1
        while out in self.used_stems:
            out = f"{stem}_dup{n}"
            n += 1
        self.used_stems.add(out)
        return out

    def write_image(
        self, img: np.ndarray, labels: List[Label], stem: str, dataset: str, group: str, source: str,
        split: Optional[str] = None,
    ) -> int:
        split = split or self.split_for(group)
        stem = self._unique_stem(stem)
        img_dir, lbl_dir = self.out / "images" / split, self.out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(img_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise IOError(f"Failed to write {img_dir / f'{stem}.jpg'}")
        (lbl_dir / f"{stem}.txt").write_text(
            "".join(f"{cls_id} {' '.join(f'{v:.6f}' for v in poly)}\n" for cls_id, poly in labels)
        )
        self.manifest.append(
            {"file": f"images/{split}/{stem}.jpg", "split": split, "dataset": dataset, "group": group,
             "source": source, "polygons": len(labels)}
        )
        self.stats[dataset][split] += 1
        self.stats[dataset]["polygons"] += len(labels)
        return 1

    def write_sample(
        self, img: np.ndarray, labels: List[Label], stem: str, dataset: str, group: str, source: str,
        split: Optional[str] = None, tile: bool = True,
    ) -> int:
        """Writes the image whole if it fits in one tile, otherwise only tiles that contain labels."""
        h, w = img.shape[:2]
        ts = self.tile_size
        if not tile or ts is None or (w <= ts and h <= ts):
            return self.write_image(img, labels, stem, dataset, group, source, split)

        px = [(cls_id, np.asarray(poly, dtype=np.float64).reshape(-1, 2) * [w, h]) for cls_id, poly in labels]
        written = 0
        for i, (tx, ty, tw, th) in enumerate(get_tile_coordinates(w, h, ts, self.overlap)):
            tile_labels: List[Label] = []
            for cls_id, pts in px:
                if pts[:, 0].max() < tx or pts[:, 0].min() > tx + tw or pts[:, 1].max() < ty or pts[:, 1].min() > ty + th:
                    continue
                tile_labels.extend((cls_id, p) for p in clip_polygon_to_tile(pts, tx, ty, tw, th))
            if not tile_labels:
                continue
            written += self.write_image(
                img[ty : ty + th, tx : tx + tw], tile_labels, f"{stem}_tile{i}", dataset, group, source, split
            )
        return written


def read_image(p: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [warn] unreadable image: {p}")
    return img


# =====================================================================
# 4. Dedicated Dataset Parsers
# =====================================================================
def ingest_coco(
    writer: BundleWriter, dataset: str, prefix: str, json_paths: Iterable[Path], image_index: Dict[str, Path],
    target_cats: set, limit: Optional[int], rng: random.Random,
) -> int:
    jobs = []
    for j_path in json_paths:
        if not j_path.exists():
            print(f"  [{dataset}] missing annotation file {j_path}")
            continue
        data = json.loads(j_path.read_text())
        cat_ids = {c["id"] for c in data.get("categories", []) if c.get("name", "").lower() in target_cats}
        images = {im["id"]: im for im in data.get("images", [])}
        anns_by_img = defaultdict(list)
        for ann in data.get("annotations", []):
            if ann.get("category_id") in cat_ids and not ann.get("iscrowd") and not ann.get("ignore"):
                anns_by_img[ann["image_id"]].append(ann)
        jobs.extend((images[i], anns) for i, anns in sorted(anns_by_img.items()) if i in images)

    count, missing = 0, 0
    for meta, anns in tqdm(cap(jobs, limit, rng), desc=dataset):
        img_p = image_index.get(Path(meta["file_name"]).name)
        if img_p is None:
            missing += 1
            continue
        img = read_image(img_p)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        labels = [(0, p) for ann in anns for poly in coco_polygons(ann) if (p := pixel_polygon_to_norm(poly, iw, ih))]
        if not labels:
            continue
        count += writer.write_sample(img, labels, f"{prefix}_{img_p.stem}", dataset, f"{prefix}_{img_p.stem}", str(img_p))
    if missing:
        print(f"  [{dataset}] {missing} annotated images not found on disk")
    return count


def ingest_ssgd(ssgd_dir: Path, writer: BundleWriter, limit: Optional[int], rng: random.Random) -> int:
    """SSGD lb101/lb201 full annotation files (fold files are re-splits of the same images)."""
    target_cats = {"crack", "broken", "scratch", "broken-membrane"}
    count = 0
    for lb in ("lb101", "lb201"):
        index = {p.name: p for p in list_images(ssgd_dir / lb)}
        json_p = ssgd_dir / f"annotations_{lb}" / f"{lb}.json"
        count += ingest_coco(writer, "SSGD", "ssgd", [json_p], index, target_cats, limit, rng)
    return count


def ingest_cardd(cardd_dir: Path, writer: BundleWriter, limit: Optional[int], rng: random.Random) -> int:
    """CarDD_COCO filtered for 'crack' and 'glass shatter'."""
    coco_dir = cardd_dir / "CarDD_COCO"
    index = {p.name: p for p in list_images(coco_dir)}
    json_paths = sorted((coco_dir / "annotations").glob("instances_*.json"))
    return ingest_coco(writer, "CarDD_release", "cardd", json_paths, index, {"crack", "glass shatter"}, limit, rng)


def ingest_agdd(agdd_dir: Path, writer: BundleWriter, limit: Optional[int], rng: random.Random) -> int:
    """AGDD: both image views share oriented-box labels in data/labels/<split>/.

    Classes: 0 contusion, 1 scratches, 2 crack, 3 spot -> keep scratches + crack.
    """
    data_dir = agdd_dir / "data"
    keep = {1: 0, 2: 0}
    jobs = []
    for view in ("images", "image"):
        view_dir = data_dir / view
        if not view_dir.is_dir():
            continue
        for split_dir in sorted(d for d in view_dir.iterdir() if d.is_dir()):
            jobs.extend((view, img_p, data_dir / "labels" / split_dir.name / f"{img_p.stem}.txt")
                        for img_p in list_images(split_dir))

    count, missing = 0, 0
    for view, img_p, lbl_p in tqdm(cap(jobs, limit, rng), desc="agdd"):
        if not lbl_p.exists():
            missing += 1
            continue
        img = read_image(img_p)
        if img is None:
            continue
        count += writer.write_sample(
            img, parse_yolo_labels(lbl_p, keep), f"agdd_{view}_{img_p.stem}", "agdd_aircraft_glass",
            f"agdd_{img_p.stem}", str(img_p),
        )
    if missing:
        print(f"  [agdd] {missing} images without label files skipped")
    return count


def ingest_crackseg9k_parquet(
    parquet_dir: Path, writer: BundleWriter, max_samples: Optional[int], rng: random.Random, batch_size: int = 128
) -> int:
    """crackseg9k Parquet: 'image' and 'mask' columns are base64-encoded PNG strings."""
    import pyarrow.parquet as pq

    p_files = sorted(parquet_dir.glob("*.parquet"))
    if not p_files:
        print("  [crackseg9k] No parquet files found.")
        return 0

    rows = [(p, i) for p in p_files for i in range(pq.ParquetFile(p).metadata.num_rows)]
    wanted = defaultdict(set)
    for p, i in cap(rows, max_samples, rng):
        wanted[p].add(i)

    def decode(value, flags) -> Optional[np.ndarray]:
        if isinstance(value, dict):
            value = value.get("bytes")
        if isinstance(value, str):
            value = base64.b64decode(value)
        if not value:
            return None
        return cv2.imdecode(np.frombuffer(value, np.uint8), flags)

    count, bad, empty = 0, 0, 0
    pbar = tqdm(total=sum(len(v) for v in wanted.values()), desc="crackseg9k")
    for p_file in p_files:
        if not wanted[p_file]:
            continue
        offset = 0
        for batch in pq.ParquetFile(p_file).iter_batches(batch_size=batch_size, columns=["image", "mask"]):
            idx = [i for i in range(batch.num_rows) if offset + i in wanted[p_file]]
            if idx:
                images, masks = batch.column("image"), batch.column("mask")
                for i in idx:
                    pbar.update(1)
                    row_id = offset + i
                    img = decode(images[i].as_py(), cv2.IMREAD_COLOR)
                    mask = decode(masks[i].as_py(), cv2.IMREAD_GRAYSCALE)
                    if img is None or mask is None or img.shape[:2] != mask.shape[:2]:
                        bad += 1
                        continue
                    polys = mask_to_yolo_polygons(mask > 127)
                    if not polys:
                        empty += 1
                        continue
                    stem = f"cs9k_{p_file.stem}_{row_id}"
                    count += writer.write_sample(img, [(0, poly) for poly in polys], stem, "crackseg9k", stem,
                                                 f"{p_file.name}#{row_id}")
            offset += batch.num_rows
    pbar.close()
    if bad or empty:
        print(f"  [crackseg9k] skipped {bad} undecodable rows and {empty} rows with empty masks")
    return count


def ingest_generic_yolo(
    src_dir: Path, writer: BundleWriter, prefix: str, dataset: str, keep: Dict[int, int],
    max_samples: Optional[int], rng: random.Random, split: Optional[str] = None, tile: bool = True,
) -> int:
    """Ultralytics-layout dataset (images/ <-> labels/); unlabeled-class images become negatives."""
    imgs = cap([p for p in list_images(src_dir) if not p.stem.endswith("_mask")], max_samples, rng)
    count, missing = 0, 0
    for img_p in tqdm(imgs, desc=dataset):
        lbl_p = yolo_label_path(img_p)
        if not lbl_p.exists():
            missing += 1
            continue
        img = read_image(img_p)
        if img is None:
            continue
        count += writer.write_sample(
            img, parse_yolo_labels(lbl_p, keep), f"{prefix}_{img_p.stem}", dataset,
            f"{prefix}_{source_key(img_p.stem)}", str(img_p), split=split, tile=tile,
        )
    if missing:
        print(f"  [{dataset}] {missing} images without label files skipped")
    return count


def ingest_cubit_negatives(
    cubit_dir: Path, writer: BundleWriter, max_tiles: int, rng: random.Random,
    tiles_per_image: int = 2, margin: int = 16,
) -> int:
    """Tiles from CUBIT images that do not touch any labeled crack or spalling polygon."""
    imgs = list_images(cubit_dir)
    rng.shuffle(imgs)
    ts = writer.tile_size
    count = 0
    pbar = tqdm(total=max_tiles, desc="neg_cubit")
    for img_p in imgs:
        if count >= max_tiles:
            break
        lbl_p = yolo_label_path(img_p)
        if not lbl_p.exists():
            continue
        img = read_image(img_p)
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        for _, poly in parse_yolo_labels(lbl_p, keep=None):
            pts = np.asarray(poly).reshape(-1, 2) * [w, h]
            boxes.append((*(pts.min(0) - margin), *(pts.max(0) + margin)))
        tiles = list(enumerate(get_tile_coordinates(w, h, ts or max(w, h), writer.overlap)))
        clean = [
            (i, t) for i, t in tiles
            if not any(x1 < t[0] + t[2] and x2 > t[0] and y1 < t[1] + t[3] and y2 > t[1] for x1, y1, x2, y2 in boxes)
        ]
        rng.shuffle(clean)
        for i, (tx, ty, tw, th) in clean[: min(tiles_per_image, max_tiles - count)]:
            count += writer.write_image(
                img[ty : ty + th, tx : tx + tw], [], f"negcubit_{img_p.stem}_tile{i}", "neg_cubit",
                f"cubit_{img_p.stem}", str(img_p),
            )
            pbar.update(1)
    pbar.close()
    return count


def ingest_empty_label_negatives(
    src_dir: Path, writer: BundleWriter, prefix: str, dataset: str, max_samples: int, rng: random.Random
) -> int:
    """Images whose YOLO label file exists and is empty (explicit background images)."""
    imgs = [p for p in list_images(src_dir) if yolo_label_path(p).exists() and not yolo_label_path(p).read_text().strip()]
    count = 0
    for img_p in tqdm(cap(imgs, max_samples, rng), desc=dataset):
        img = read_image(img_p)
        if img is not None:
            count += writer.write_sample(img, [], f"{prefix}_{img_p.stem}", dataset,
                                         f"{prefix}_{source_key(img_p.stem)}", str(img_p))
    return count


def ingest_empty_mask_negatives(
    src_dir: Path, writer: BundleWriter, prefix: str, dataset: str, max_samples: int, rng: random.Random
) -> int:
    """Images whose '<stem>_mask.png' semantic mask has no foreground pixels."""
    imgs = []
    for p in list_images(src_dir):
        mask_p = p.with_name(f"{p.stem}_mask.png")
        if p.stem.endswith("_mask") or not mask_p.exists():
            continue
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if mask is not None and not mask.any():
            imgs.append(p)
    count = 0
    for img_p in tqdm(cap(imgs, max_samples, rng), desc=dataset):
        img = read_image(img_p)
        if img is not None:
            count += writer.write_sample(img, [], f"{prefix}_{img_p.stem}", dataset,
                                         f"{prefix}_{source_key(img_p.stem)}", str(img_p))
    return count


# =====================================================================
# 5. Main Assembly Loop
# =====================================================================
def prepare_out_dir(out: Path, data_root: Path) -> None:
    out_r, root_r = out.resolve(), data_root.resolve()
    if out_r == root_r or out_r in root_r.parents:
        sys.exit(f"--out-dir {out} must not be the data root or one of its parents")
    if out.exists() and any(out.iterdir()):
        unknown = sorted(p.name for p in out.iterdir() if p.name not in BUNDLE_ENTRIES)
        if unknown:
            sys.exit(f"{out} exists and contains non-bundle entries {unknown}; refusing to overwrite")
        print(f"Removing previous bundle at {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare and harmonize Glass Crack Datasets.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Path to raw datasets directory")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed_bundle_v1"), help="Target bundle output")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile resolution (preserves hairline cracks)")
    parser.add_argument("--overlap", type=float, default=0.20, help="Tile overlap ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", nargs="*", default=["agdd"], choices=DATASET_KEYS,
                        help="Datasets to skip (default: agdd, the external test set)")
    parser.add_argument("--limit", type=int, default=None, help="Cap source images per dataset (smoke runs)")
    parser.add_argument("--max-ucrack", type=int, default=1500)
    parser.add_argument("--max-cs9k", type=int, default=2500)
    parser.add_argument("--max-neg-cubit", type=int, default=600, help="Clean CUBIT tiles")
    parser.add_argument("--max-neg-gseg", type=int, default=600)
    parser.add_argument("--max-neg-fcd", type=int, default=250)
    args = parser.parse_args()

    if not 0.0 <= args.val_ratio < 1.0:
        sys.exit("--val-ratio must be in [0, 1)")
    if not 0.0 <= args.overlap < 1.0:
        sys.exit("--overlap must be in [0, 1)")

    root, out = args.data_root, args.out_dir
    if not root.is_dir():
        sys.exit(f"data root {root} does not exist")
    prepare_out_dir(out, root)
    writer = BundleWriter(out, args.val_ratio, args.tile_size, args.overlap)

    def lim(n: Optional[int]) -> Optional[int]:
        if args.limit is None:
            return n
        return args.limit if n is None else min(n, args.limit)

    print("\n=======================================================")
    print(f" Processing Glass Crack Datasets from: {root.resolve()}")
    print(f" Target Output: {out.resolve()}")
    print(f" Tile Resolution: {args.tile_size}px (with {args.overlap:.0%} overlap)")
    print("=======================================================\n")

    steps = [
        ("gdd2", "glass_defect_detection2", "glass_defect_detection2 (crack + scratch)",
         lambda d, r: ingest_generic_yolo(d, writer, "gdd2", "glass_defect_detection2", {1: 0, 3: 0}, lim(None), r)),
        ("ssgd", "SSGD", "SSGD", lambda d, r: ingest_ssgd(d, writer, lim(None), r)),
        ("agdd", "agdd_aircraft_glass", "agdd_aircraft_glass (scratches + crack)",
         lambda d, r: ingest_agdd(d, writer, lim(None), r)),
        ("cardd", "CarDD_release", "CarDD_release (glass shatter & crack)",
         lambda d, r: ingest_cardd(d, writer, lim(None), r)),
        ("ucrack", "crackSegUltralytics", f"crackSegUltralytics (subsampled to {lim(args.max_ucrack)})",
         lambda d, r: ingest_generic_yolo(d, writer, "ucrack", "crackSegUltralytics", {0: 0}, lim(args.max_ucrack), r)),
        ("cs9k", "crackseg9k", f"crackseg9k Parquet (subsampled to {lim(args.max_cs9k)})",
         lambda d, r: ingest_crackseg9k_parquet(d, writer, lim(args.max_cs9k), r)),
        ("neg_cubit", "cubit_inseg", f"clean CUBIT negative tiles ({lim(args.max_neg_cubit)})",
         lambda d, r: ingest_cubit_negatives(d, writer, lim(args.max_neg_cubit), r)),
        ("neg_gseg", "glass_segmentation_2026", "empty-label glass_segmentation_2026 negatives",
         lambda d, r: ingest_empty_label_negatives(d, writer, "neggseg", "neg_glass2026", lim(args.max_neg_gseg), r)),
        ("neg_fcd", "fcd_108_facade", "empty-mask fcd_108_facade negatives",
         lambda d, r: ingest_empty_mask_negatives(d, writer, "negfcd", "neg_fcd", lim(args.max_neg_fcd), r)),
        ("ncugd", "ncugd_glass_crack", "ncugd_glass_crack as isolated external test benchmark",
         lambda d, r: ingest_generic_yolo(d, writer, "ncugd", "ncugd_glass_crack", {0: 0}, None, r,
                                          split=EXTERNAL_SPLIT, tile=False)),
    ]

    if "agdd" not in args.exclude:
        print("WARNING: AGDD is included in train/val; it can no longer serve as an external test set.\n")

    counts: Dict[str, int] = {}
    for n, (key, folder, title, fn) in enumerate(steps, 1):
        if key in args.exclude:
            print(f"--> [{n}/{len(steps)}] Skipping {folder} (excluded)")
            continue
        src = root / folder
        if not src.is_dir():
            print(f"--> [{n}/{len(steps)}] Skipping {folder} (not found at {src})")
            continue
        print(f"--> [{n}/{len(steps)}] Ingesting {title}...")
        counts[key] = fn(src, random.Random(f"{args.seed}:{key}"))

    if not any(writer.stats[d]["train"] for d in writer.stats):
        sys.exit("No training samples were written; check --data-root")

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"test: images/{EXTERNAL_SPLIT}\n"
        "\n"
        "names:\n"
        "  0: crack\n"
    )

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "split", "dataset", "group", "source", "polygons"])
        w.writeheader()
        w.writerows(writer.manifest)

    totals = Counter()
    for c in writer.stats.values():
        totals.update({k: v for k, v in c.items() if k != "polygons"})
    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "datasets": {d: dict(c) for d, c in writer.stats.items()},
        "totals": dict(totals),
    }
    (out / "summary_report.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Dataset Preparation Complete ===")
    print(f"YAML configuration: {out / 'data.yaml'}")
    print(f"Summary report: {out / 'summary_report.json'}")
    for d, c in writer.stats.items():
        splits = ", ".join(f"{s}={c[s]}" for s in ("train", "val", EXTERNAL_SPLIT) if c[s])
        print(f"  * {d:25}: {splits}  ({c['polygons']} polygons)")
    print(f"  = totals: {dict(totals)}")


if __name__ == "__main__":
    main()
