"""Track A: YOLO-seg training (E1, E2, ...) with resume and an evaluation hook.

    python -m glass_crack_models.train.train_yolo --config e2_yolo_linear.yaml [--seed 1]
    python -m glass_crack_models.train.train_yolo --config e2_yolo_linear.yaml --smoke    # laptop CPU check

Runs land in OUT_ROOT/runs/<exp_id>_s<seed>/. Re-running the same command after a Colab disconnect
resumes from weights/last.pt. When training ends, run_eval scores weights/best.pt and appends to results.csv.

Optional config keys (plan v4):
  data:            a data.yaml (``${DATA_ROOT}``/``${OUT_ROOT}`` allowed) used instead of `dataset` (proxy pretraining)
  hard_negatives:  {dir: <images dir>, frac: 0.3}: crack-free images added so they make up `frac` of the train list
  albu:            Albumentations pixel transforms, e.g. [{MotionBlur: {blur_limit: [3, 15], p: 0.3}}]
  eval:            false to skip run_eval (pretraining runs, whose classes differ from the glass set)
`model` may be a checkpoint path (``${OUT_ROOT}/runs/E5pre_s0/weights/best.pt``) for fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import yaml

from glass_crack_models import paths
from glass_crack_models.config import load_config
from glass_crack_models.protocol import spread

DATASETS = {"linear": paths.linear_dir, "merged": paths.merged_dir}


def data_yaml(cfg: dict, run_dir: Path) -> Path:
    if "data" in cfg:
        return paths.resolve(cfg["data"])
    if "dataset" not in cfg:
        hint = ("this looks like a Track B / U-Net config — use "
                "`python -m glass_crack_models.train.train_semseg --config ...` instead"
                if cfg.get("track") == "B" or "arch" in cfg else
                "set `dataset: merged` or `dataset: linear`, or pass `data: <data.yaml>`")
        raise KeyError(f"config missing 'dataset'; {hint}")
    ds = DATASETS[cfg["dataset"]]()
    y = ds / f"data_{cfg.get('weighting', 'balanced')}.yaml"
    if not y.exists():
        raise FileNotFoundError(f"{y} missing: run `python -m glass_crack_models.data.relabel_classes` first")
    hn = cfg.get("hard_negatives")
    return with_hard_negatives(y, paths.resolve(hn["dir"]), float(hn["frac"]), run_dir) if hn else y


def with_hard_negatives(y: Path, neg_dir: Path, frac: float, run_dir: Path) -> Path:
    """Copy of data yaml `y` whose train list also holds crack-free images (a fixed sample, seed 0) from `neg_dir`."""
    d = yaml.safe_load(y.read_text())
    ds = Path(d["path"])
    train = ds / d["train"]
    core = ([str(ds / ln.removeprefix("./")) for ln in train.read_text().split()] if train.suffix == ".txt"
            else sorted(str(p) for p in train.iterdir()))
    negs = sorted(str(p) for p in neg_dir.glob("*.jpg"))
    k = min(len(negs), round(len(core) * frac / (1 - frac)))
    lst = run_dir / "train_hard_negatives.txt"
    run_dir.mkdir(parents=True, exist_ok=True)
    lst.write_text("\n".join(core + random.Random(0).sample(negs, k)) + "\n")
    print(f"hard negatives: {k} of {len(negs)} added to {len(core)} train entries ({k / (k + len(core)):.0%})")
    out = run_dir / "data_hard_negatives.yaml"
    out.write_text(yaml.safe_dump({**d, "train": str(lst)}))
    return out


def albumentations(specs: list[dict]) -> list:
    import albumentations as A
    return [getattr(A, name)(**kw) for spec in specs for name, kw in spec.items()]


def smoke_yaml(cfg: dict, run_dir: Path, n: int = 64) -> Path:
    """Evenly spaced train subset (sorted order puts the mostly bubble/inclusion GDD2 images first)."""
    ds = DATASETS[cfg["dataset"]]()
    imgs = sorted((ds / "train" / "images").iterdir())
    pick = spread(imgs, min(n, len(imgs)))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "smoke_train.txt").write_text("\n".join(str(p) for p in pick) + "\n")
    names = yaml.safe_load((ds / "data_all.yaml").read_text())["names"]
    y = run_dir / "smoke_data.yaml"
    y.write_text(yaml.safe_dump({"path": str(ds), "train": str(run_dir / "smoke_train.txt"),
                                 "val": "valid/images", "test": "test/images", "nc": len(names), "names": names}))
    return y


def finished(last: Path) -> bool:
    ck = torch.load(last, map_location="cpu", weights_only=False)
    return ck.get("epoch", -1) == -1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--weighting", default=None, help="Override config: all | balanced | no_augpairs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 5%% of train, imgsz 320, CPU; eval on 12 images")
    ap.add_argument("--epochs", type=int, default=None, help="Override config (and the smoke default of 1)")
    ap.add_argument("--fresh", action="store_true", help="Ignore an existing last.pt")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--workers", type=int, default=None, help="Override config dataloader workers")
    ap.add_argument("--compute-units", default="", help="Compute used, e.g. Colab units or the cluster GPU (logged to results.csv)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.weighting:
        cfg["weighting"] = args.weighting
    seed = int(cfg.get("seed", 0))
    name = f"{cfg['exp_id']}_s{seed}" + ("_smoke" if args.smoke else "")
    project = paths.out_root() / "runs"
    run_dir = project / name
    last, best = run_dir / "weights" / "last.pt", run_dir / "weights" / "best.pt"

    from ultralytics import YOLO
    train_kw = dict(cfg.get("train", {}))
    if cfg.get("albu"):
        train_kw["augmentations"] = albumentations(cfg["albu"])
    train_kw.update(data=str(data_yaml(cfg, run_dir)), project=str(project), name=name, exist_ok=True, seed=seed, deterministic=True)
    if args.device:
        train_kw["device"] = args.device
    if args.workers is not None:
        train_kw["workers"] = args.workers
    if args.smoke:
        train_kw.update(data=str(smoke_yaml(cfg, run_dir)), epochs=1, imgsz=320, batch=4, workers=0,
                        device=args.device or "cpu", plots=False, patience=0)
    if args.epochs:
        train_kw["epochs"] = args.epochs

    if last.exists() and not args.fresh and finished(last):
        print(f"{name}: training already finished ({last})")
    elif last.exists() and not args.fresh:
        print(f"{name}: resuming from {last}")
        YOLO(str(last)).train(resume=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "experiment.json").write_text(json.dumps({k: v for k, v in cfg.items()}, indent=1, default=str))
        YOLO(str(paths.resolve(cfg["model"]))).train(**train_kw)

    if args.no_eval or not cfg.get("eval", True):
        return
    from glass_crack_models.eval import run_eval
    ev = ["--checkpoint", str(best), "--exp-id", cfg["exp_id"], "--seed", str(seed),
          "--task", Path(cfg["model"]).stem, "--pretrain", cfg.get("pretrain", ""), "--weighting", cfg.get("weighting", ""),
          "--compute-units", args.compute_units, "--notes", cfg.get("notes", "")]
    if args.device:
        ev += ["--device", args.device]
    if args.smoke:
        ev += ["--limit", "12", "--bootstrap", "200", "--out", str(paths.out_root() / "evals" / name)]
    run_eval.main(ev)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
