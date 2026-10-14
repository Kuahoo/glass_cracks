"""Build the linear_defect dataset and the source-weighted train lists.

    python -m glass_crack_models.data.relabel_classes

Writes, under DATA_ROOT/processed/:
  glass_merged_linear/            crack + scratch -> class 0 `linear_defect`; bubble/inclusion rows
                                  dropped, their images kept as negatives
  <dataset>/train_balanced.txt    single-copy photos listed twice, augmented-pair copies once, so every
                                  source photo carries equal weight (plan Section 5, Phase 4)
  <dataset>/train_no_augpairs.txt single-copy photos only (Phase 4 ablation)
  <dataset>/data_{all,balanced,no_augpairs}.yaml
for both glass_merged_seg (4-class) and glass_merged_linear.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

from glass_crack_models import paths
from glass_crack_models.protocol import LINEAR_NAMES, LINEAR_SOURCE_IDS, MERGED_NAMES

SPLITS = ("train", "valid", "test")
WEIGHTINGS = ("all", "balanced", "no_augpairs")


def relabel_lines(lines: list[str], keep_ids: set[int], new_id: int = 0) -> list[str]:
    out = []
    for line in lines:
        t = line.split()
        if t and int(float(t[0])) in keep_ids:
            out.append(" ".join([str(new_id)] + t[1:]))
    return out


def _link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_linear(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    for split in SPLITS:
        (dst / split / "images").mkdir(parents=True)
        (dst / split / "labels").mkdir(parents=True)
        for img in sorted((src / split / "images").iterdir()):
            _link_or_copy(img, dst / split / "images" / img.name)
            lf = src / split / "labels" / (img.stem + ".txt")
            lines = lf.read_text(encoding="utf-8").splitlines() if lf.exists() else []
            new = relabel_lines(lines, LINEAR_SOURCE_IDS)
            (dst / split / "labels" / lf.name).write_text("\n".join(new) + ("\n" if new else ""), encoding="utf-8")
    shutil.copy2(src / "manifest.csv", dst / "manifest.csv")


def write_weighting_lists(ds: Path, names: list[str]):
    with (ds / "manifest.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    train = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["image"])
    pair = [r for r in train if r["copy_kind"] == "original_or_augmented"]
    single = [r for r in train if r["copy_kind"] != "original_or_augmented"]
    rel = lambda r: f"./train/images/{r['image']}"
    lists = {
        "balanced": [rel(r) for r in single for _ in range(2)] + [rel(r) for r in pair],
        "no_augpairs": [rel(r) for r in single],
    }
    for name, items in lists.items():
        (ds / f"train_{name}.txt").write_text("\n".join(sorted(items)) + "\n", encoding="utf-8")
    for w in WEIGHTINGS:
        train_ref = "train/images" if w == "all" else f"train_{w}.txt"
        (ds / f"data_{w}.yaml").write_text(
            f"# source weighting: {w}\n"
            f"path: {ds}\n"
            f"train: {train_ref}\nval: valid/images\ntest: test/images\n"
            f"nc: {len(names)}\nnames: [{', '.join(names)}]\n", encoding="utf-8")
    return {"train_images": len(train), "augpair_images": len(pair), "single_images": len(single)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args(argv)
    merged, linear = paths.merged_dir(), paths.linear_dir()
    build_linear(merged, linear)
    print("4-class  ", write_weighting_lists(merged, MERGED_NAMES))
    print("linear   ", write_weighting_lists(linear, LINEAR_NAMES))
    print("wrote", linear)


if __name__ == "__main__":
    main()
