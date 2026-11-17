"""Track C: RF-DETR-Seg training (E7) with resume and the run_eval hook.

    python -m glass_crack_models.train.train_rfdetr --config e7_rfdetr_seg.yaml [--seed 1]
    python -m glass_crack_models.train.train_rfdetr --config e7_rfdetr_seg.yaml --smoke    # laptop CPU check

Runs land in OUT_ROOT/runs/<exp_id>_s<seed>/ (RF-DETR's output_dir) with experiment.json (`track: C`).
RF-DETR reads a YOLO folder, not Ultralytics train lists, so the train split is materialized under
<run>/dataset/ from the same `weighting` list Track A uses (balanced: single-copy photos twice, augmented
pair copies once) plus valid; test is left out so training cannot touch it.
Re-running the same command resumes from last.ckpt, or only re-scores if training already finished.
When training ends, run_eval scores the preferred checkpoint (best_total, then ema, then regular).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

from glass_crack_models import paths
from glass_crack_models.config import load_config
from glass_crack_models.data.relabel_classes import _link_or_copy
from glass_crack_models.infer.predictors import prefer_rfdetr_checkpoint, torch_device

DATASETS = {"linear": paths.linear_dir, "merged": paths.merged_dir}
VARIANTS = {"nano": "RFDETRSegNano", "small": "RFDETRSegSmall", "medium": "RFDETRSegMedium", "large": "RFDETRSegLarge"}
BEST = ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth")
SMOKE_TRAIN, SMOKE_VALID = 16, 8


def variant_class(name: str):
    import rfdetr
    key = name.lower().strip()
    if key not in VARIANTS:
        raise KeyError(f"unknown RF-DETR variant {name!r}; choose from {sorted(VARIANTS)}")
    return getattr(rfdetr, VARIANTS[key])


def spread(items: list, n: int) -> list:
    """`n` items evenly spaced over a sorted list (sorted order clusters by source dataset)."""
    return items if n >= len(items) else [items[int(i * len(items) / n)] for i in range(n)]


def train_images(src: Path, weighting: str) -> list[Path]:
    """Train images with repeats, exactly as Track A's data_<weighting>.yaml lists them."""
    if weighting == "all":
        return sorted((src / "train" / "images").iterdir())
    lst = src / f"train_{weighting}.txt"
    if not lst.exists():
        raise FileNotFoundError(f"{lst} missing: run `python -m glass_crack_models.data.relabel_classes` first")
    return [src / ln.removeprefix("./") for ln in lst.read_text().splitlines() if ln.strip()]


def build_dataset(cfg: dict, out: Path, smoke: bool = False) -> Path:
    """YOLO folder for RF-DETR: <out>/{train,valid}/{images,labels} + data.yaml (hard links, copies as fallback)."""
    src = DATASETS[cfg["dataset"]]()
    train = train_images(src, cfg.get("weighting", "balanced"))
    valid = sorted((src / "valid" / "images").iterdir())
    if smoke:
        train, valid = spread(sorted(set(train)), SMOKE_TRAIN), spread(valid, SMOKE_VALID)
    shutil.rmtree(out, ignore_errors=True)
    for split, imgs in (("train", train), ("valid", valid)):
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        seen = Counter()
        for img in imgs:
            k = seen[img.name]
            seen[img.name] += 1
            stem = img.stem + (f"_w{k}" if k else "")  # a repeat needs its own file name
            _link_or_copy(img, out / split / "images" / (stem + img.suffix))
            lbl = img.parent.parent / "labels" / (img.stem + ".txt")
            (out / split / "labels" / (stem + ".txt")).write_text(lbl.read_text() if lbl.exists() else "")
    names = yaml.safe_load((src / "data_all.yaml").read_text())["names"]
    (out / "data.yaml").write_text(yaml.safe_dump({"train": "train/images", "val": "valid/images",
                                                   "nc": len(names), "names": list(names)}))
    print(f"dataset: {len(train)} train entries ({len(set(train))} images), {len(valid)} valid -> {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None, help="cpu | cuda | cuda:0 | 0 (default: RF-DETR picks)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"1 epoch on {SMOKE_TRAIN} train / {SMOKE_VALID} valid images, batch 1; eval on 12 images")
    ap.add_argument("--epochs", type=int, default=None, help="Override config (and the smoke default of 1)")
    ap.add_argument("--fresh", action="store_true", help="Discard this run's checkpoints and start over")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--workers", type=int, default=None, help="Override config dataloader workers")
    ap.add_argument("--compute-units", default="")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = int(cfg.get("seed", 0))
    variant = cfg.get("variant", "large")
    name = f"{cfg['exp_id']}_s{seed}" + ("_smoke" if args.smoke else "")
    run_dir = paths.out_root() / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    last, done = run_dir / "last.ckpt", run_dir / "train_done.json"
    if args.fresh:
        for f in (last, done, *(run_dir / b for b in BEST)):
            f.unlink(missing_ok=True)

    if done.exists():
        print(f"{name}: training already finished ({done})")
    else:
        train_kw = dict(
            output_dir=str(run_dir), seed=seed, lr=float(cfg.get("lr", 1e-4)),
            epochs=int(cfg.get("epochs", 50)), batch_size=int(cfg.get("batch_size", 2)),
            grad_accum_steps=int(cfg.get("grad_accum_steps", 8)), num_workers=int(cfg.get("workers", 8)),
        )
        if args.workers is not None:
            train_kw["num_workers"] = args.workers
        if args.smoke:
            train_kw.update(epochs=1, batch_size=1, grad_accum_steps=1, num_workers=0)
        if args.epochs:
            train_kw["epochs"] = args.epochs
        if args.device:
            train_kw["device"] = torch_device(args.device)
        if last.exists():  # same dataset folder as the interrupted run
            train_kw["resume"] = str(last)
            print(f"{name}: resuming from {last}")
        else:
            build_dataset(cfg, run_dir / "dataset", args.smoke)
            (run_dir / "experiment.json").write_text(json.dumps({**cfg, "track": "C"}, indent=1, default=str))
        variant_class(variant)().train(dataset_dir=str(run_dir / "dataset"), **train_kw)
        done.write_text(json.dumps({"epochs": train_kw["epochs"], "checkpoint": prefer_rfdetr_checkpoint(run_dir).name}))

    if args.no_eval:
        return
    from glass_crack_models.eval import run_eval
    ev = ["--checkpoint", str(prefer_rfdetr_checkpoint(run_dir)), "--exp-id", cfg["exp_id"], "--seed", str(seed),
          "--task", f"rfdetr-seg-{variant}", "--pretrain", "coco", "--weighting", cfg.get("weighting", ""),
          "--compute-units", args.compute_units, "--notes", cfg.get("notes", "")]
    if args.device:
        ev += ["--device", args.device]
    if args.smoke:
        ev += ["--limit", "12", "--bootstrap", "200", "--out", str(paths.out_root() / "evals" / name)]
    run_eval.main(ev)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
