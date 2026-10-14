"""Overlays and galleries.

GT overlay check (Phase 0), ~20 images per split including empty-label ones:
    python -m glass_crack_models.viz.overlay gt --dataset merged --n 20
    python -m glass_crack_models.viz.overlay gt --dataset linear --n 20
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from glass_crack_models import paths
from glass_crack_models.data.rasterize import image_paths, label_path_for, read_yolo
from glass_crack_models.protocol import LINEAR_NAMES, MERGED_NAMES

# RGB
CLASS_COLORS = [(0, 200, 255), (255, 60, 60), (255, 200, 0), (60, 255, 60), (200, 80, 255)]
PRED_COLOR = (255, 0, 255)
GT_COLOR = (0, 255, 0)


def draw_polygons(img: np.ndarray, labels, names=None, thickness: int = 2) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for c, poly in labels:
        color = CLASS_COLORS[c % len(CLASS_COLORS)]
        pts = np.round(poly * [w, h]).astype(np.int32)
        cv2.polylines(out, [pts], True, color, thickness)
        if names:
            x, y = pts[:, 0].min(), max(pts[:, 1].min() - 4, 10)
            cv2.putText(out, names[c] if c < len(names) else str(c), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
    return out


def draw_mask(img: np.ndarray, mask: np.ndarray, color=PRED_COLOR, alpha: float = 0.45, contour: bool = True) -> np.ndarray:
    out = img.copy()
    if mask.any():
        out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        if contour:
            cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cs, -1, color, max(1, round(max(img.shape[:2]) / 800)))
    return out


def pred_overlay(img: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray | None = None) -> np.ndarray:
    """Prediction filled in magenta; GT (if given) as green outline."""
    out = draw_mask(img, pred_mask)
    if gt_mask is not None and gt_mask.any():
        cs, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, GT_COLOR, 1)
    return out


def caption(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((22, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text[:80], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def grid(images: list[np.ndarray], cols: int = 5, cell: int = 320) -> np.ndarray:
    if not images:
        return np.full((cell, cell, 3), 255, np.uint8)
    cells = []
    for im in images:
        s = cell / max(im.shape[:2])
        im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
        pad = np.full((cell + 22, cell, 3), 255, np.uint8)
        pad[:im.shape[0], :im.shape[1]] = im
        cells.append(pad)
    while len(cells) % cols:
        cells.append(np.full_like(cells[0], 255))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)


def save_rgb(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def read_rgb(path: Path | str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def gt_overlays(dataset_dir: Path, names: list[str], out_dir: Path, n: int = 20, n_empty: int = 4, seed: int = 0):
    rng = random.Random(seed)
    for split in ("train", "valid", "test"):
        imgs = image_paths(dataset_dir / split)
        labeled = [p for p in imgs if read_yolo(label_path_for(p))]
        empty = [p for p in imgs if not read_yolo(label_path_for(p))]
        pick = rng.sample(labeled, min(n - n_empty, len(labeled))) + rng.sample(empty, min(n_empty, len(empty)))
        tiles = []
        for p in pick:
            labels = read_yolo(label_path_for(p))
            img = draw_polygons(read_rgb(p), labels, names)
            tiles.append(caption(img, f"{p.name} ({len(labels)} polys)"))
            save_rgb(out_dir / split / (p.stem + ".jpg"), img)
        save_rgb(out_dir / f"{split}_contact_sheet.jpg", grid(tiles, cols=5))
        print(f"{split}: {len(pick)} overlays ({min(n_empty, len(empty))} empty-label) -> {out_dir / split}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gt", help="Render GT polygons for a sample of each split")
    g.add_argument("--dataset", choices=["merged", "linear"], default="merged")
    g.add_argument("--n", type=int, default=20)
    g.add_argument("--n-empty", type=int, default=4)
    g.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "gt":
        ds, names = (paths.merged_dir(), MERGED_NAMES) if args.dataset == "merged" else (paths.linear_dir(), LINEAR_NAMES)
        out = args.out or paths.out_root() / "overlays" / f"gt_{args.dataset}"
        gt_overlays(ds, names, out, args.n, args.n_empty)


if __name__ == "__main__":
    main()
