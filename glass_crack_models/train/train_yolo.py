"""Track A: YOLO-seg training (E1, E2, ...) with resume and an evaluation hook.

    python -m glass_crack_models.train.train_yolo --config e2_yolo_linear.yaml [--seed 1]
    python -m glass_crack_models.train.train_yolo --config e2_yolo_linear.yaml --smoke    # laptop CPU check

Runs land in OUT_ROOT/runs/<exp_id>_s<seed>/. Re-running the same command after a Colab disconnect
resumes from weights/last.pt. When training ends, run_eval scores weights/best.pt and appends to results.csv.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from glass_crack_models import paths
from glass_crack_models.config import load_config

DATASETS = {"linear": paths.linear_dir, "merged": paths.merged_dir}


def data_yaml(cfg: dict) -> Path:
    ds = DATASETS[cfg["dataset"]]()
    y = ds / f"data_{cfg.get('weighting', 'balanced')}.yaml"
    if not y.exists():
        raise FileNotFoundError(f"{y} missing: run `python -m glass_crack_models.data.relabel_classes` first")
    return y


def smoke_yaml(cfg: dict, run_dir: Path, n: int = 64) -> Path:
    """Evenly spaced train subset (sorted order puts the mostly bubble/inclusion GDD2 images first)."""
    ds = DATASETS[cfg["dataset"]]()
    imgs = sorted((ds / "train" / "images").iterdir())
    pick = [imgs[int(i * len(imgs) / n)] for i in range(min(n, len(imgs)))]
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
    ap.add_argument("--compute-units", default="", help="Colab compute units used (logged to results.csv)")
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
    train_kw.update(data=str(data_yaml(cfg)), project=str(project), name=name, exist_ok=True, seed=seed, deterministic=True)
    if args.device:
        train_kw["device"] = args.device
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
        YOLO(cfg["model"]).train(**train_kw)

    if args.no_eval:
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
