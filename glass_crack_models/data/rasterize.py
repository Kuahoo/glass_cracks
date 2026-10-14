"""YOLO segmentation labels <-> pixel masks.

CLI: write binary PNG masks next to a YOLO-seg split (optional; Track B rasterizes on the fly).
    python -m glass_crack_models.data.rasterize --split-dir <dataset>/train --classes 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_yolo(path: Path | str) -> list[tuple[int, np.ndarray]]:
    """Return [(class_id, Nx2 normalized polygon)] from a YOLO-seg label file (missing file -> [])."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.split()
        if len(t) < 7:
            continue
        out.append((int(float(t[0])), np.array(t[1:], dtype=np.float64).reshape(-1, 2)))
    return out


def polygon_to_mask(poly: np.ndarray, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    pts = np.round(poly * [w, h]).astype(np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def instance_masks(labels, h: int, w: int, classes: set[int] | None = None) -> list[np.ndarray]:
    return [polygon_to_mask(p, h, w) for c, p in labels if classes is None or c in classes]


def semantic_mask(labels, h: int, w: int, classes: set[int] | None = None) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for c, p in labels:
        if classes is None or c in classes:
            cv2.fillPoly(m, [np.round(p * [w, h]).astype(np.int32)], 1)
    return m.astype(bool)


def mask_to_polygons(mask: np.ndarray, epsilon: float = 0.5, min_points: int = 3) -> list[np.ndarray]:
    """Outer contours of a binary mask as pixel-coordinate polygons (Nx2 float)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in contours:
        c = cv2.approxPolyDP(c, epsilon, True) if epsilon > 0 else c
        if len(c) >= min_points:
            polys.append(c.reshape(-1, 2).astype(np.float64))
    return polys


def label_path_for(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")


def image_paths(split_dir: Path) -> list[Path]:
    return sorted(p for p in (split_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-dir", required=True, type=Path)
    ap.add_argument("--classes", default="", help="Comma-separated class ids to include (default: all)")
    args = ap.parse_args(argv)
    classes = {int(c) for c in args.classes.split(",") if c != ""} or None
    out = args.split_dir / "masks"
    out.mkdir(exist_ok=True)
    for img in image_paths(args.split_dir):
        h, w = cv2.imread(str(img), cv2.IMREAD_UNCHANGED).shape[:2]
        m = semantic_mask(read_yolo(label_path_for(img)), h, w, classes)
        cv2.imwrite(str(out / (img.stem + ".png")), m.astype(np.uint8) * 255)
    print("wrote", out)


if __name__ == "__main__":
    main()
