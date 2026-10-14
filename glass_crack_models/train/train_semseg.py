"""Track B: semantic segmentation (U-Net / SegFormer via segmentation-models-pytorch), E3 and ablations.

    python -m glass_crack_models.train.train_semseg --config e3_unet_r34.yaml [--seed 1]
    python -m glass_crack_models.train.train_semseg --config e3_unet_r34.yaml --smoke    # laptop CPU check

Checkpoints: OUT_ROOT/runs/<exp_id>_s<seed>/{last.pth (full state, every epoch), best.pth (best valid Dice)}.
Re-running the same command resumes from last.pth. Checkpoint selection uses valid only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from glass_crack_models import paths
from glass_crack_models.config import load_config
from glass_crack_models.data.rasterize import image_paths
from glass_crack_models.infer.predictors import torch_device
from glass_crack_models.train.semseg_common import (SegDataset, build_augment, build_model, make_loss,
                                                    train_images_and_weights)

CLASSES = ["linear_defect"]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def lr_lambda(warmup: int, total: int):
    def f(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        t = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f


@torch.no_grad()
def validate(model, loader, loss_fn, device, amp):
    model.eval()
    tp = fp = fn = 0.0
    losses = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp):
            logits = model(x)
        logits = logits.float()
        losses.append(loss_fn(logits, y).item())
        p = (torch.sigmoid(logits) >= 0.5).float()
        tp += (p * y).sum().item()
        fp += (p * (1 - y)).sum().item()
        fn += ((1 - p) * y).sum().item()
    dice = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0
    return float(np.mean(losses)) if losses else float("nan"), dice


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--weighting", default=None, help="Override config: all | balanced | no_augpairs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 16 train / 8 valid images at 320 px; eval on 12 images")
    ap.add_argument("--epochs", type=int, default=None, help="Override config (and the smoke default of 1)")
    ap.add_argument("--fresh", action="store_true", help="Ignore an existing last.pth")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--compute-units", default="")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.weighting:
        cfg["weighting"] = args.weighting
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = torch_device(args.device)
    amp = bool(cfg.get("amp", True)) and device.startswith("cuda")
    name = f"{cfg['exp_id']}_s{seed}" + ("_smoke" if args.smoke else "")
    run_dir = paths.out_root() / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_p, best_p, hist_p = run_dir / "last.pth", run_dir / "best.pth", run_dir / "history.csv"

    data_dir = paths.linear_dir()
    if not (data_dir / "manifest.csv").exists():
        raise FileNotFoundError(f"{data_dir} missing: run `python -m glass_crack_models.data.relabel_classes` first")
    imgsz, epochs, batch, workers = int(cfg["imgsz"]), int(cfg["epochs"]), int(cfg["batch"]), int(cfg.get("workers", 4))
    train_imgs, weights = train_images_and_weights(data_dir, cfg.get("weighting", "balanced"))
    valid_imgs = image_paths(data_dir / "valid")
    if args.smoke:
        from glass_crack_models.eval.run_eval import spread
        keep = set(spread(list(range(len(train_imgs))), 16))
        train_imgs = [p for i, p in enumerate(train_imgs) if i in keep]
        weights = [w for i, w in enumerate(weights) if i in keep]
        valid_imgs = spread(valid_imgs, 8)
        imgsz, epochs, batch, workers = 320, 1, 2, 0
    if args.epochs:
        epochs = args.epochs

    g = torch.Generator().manual_seed(seed)
    train_ds = SegDataset(train_imgs, {0}, imgsz, build_augment(cfg.get("aug", "base"), imgsz))
    sampler = WeightedRandomSampler(weights, num_samples=len(train_imgs), replacement=True, generator=g)
    train_dl = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=workers, drop_last=True,
                          pin_memory=device.startswith("cuda"), persistent_workers=workers > 0,
                          worker_init_fn=lambda w: np.random.seed(seed * 1000 + w))
    val_dl = DataLoader(SegDataset(valid_imgs, {0}, imgsz), batch_size=batch, num_workers=workers,
                        pin_memory=device.startswith("cuda"))

    model_cfg = {k: cfg[k] for k in ("arch", "encoder")}
    model = build_model(cfg["arch"], cfg["encoder"], cfg.get("encoder_weights"), len(CLASSES)).to(device)
    loss_fn = make_loss(cfg.get("loss", "bce_dice"), cfg.get("pos_weight"))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(int(cfg.get("warmup_epochs", 2)), epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, stale = 0, -1.0, 0

    if last_p.exists() and not args.fresh:
        st = torch.load(last_p, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        start, best, stale = st["epoch"] + 1, st["best"], st["stale"]
        torch.set_rng_state(st["rng_cpu"])
        g.set_state(st["rng_sampler"])
        print(f"{name}: resumed at epoch {start} (best valid Dice {best:.4f})")
    else:
        (run_dir / "experiment.json").write_text(json.dumps(cfg, indent=1, default=str))

    patience = int(cfg.get("patience", 25))
    for epoch in range(start, epochs):
        if stale >= patience:
            print(f"early stop: no valid improvement for {patience} epochs")
            break
        model.train()
        t0, losses = time.time(), []
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        lr = opt.param_groups[0]["lr"]
        sched.step()
        val_loss, dice = validate(model, val_dl, loss_fn, device, amp)
        improved = dice > best
        best, stale = (dice, 0) if improved else (best, stale + 1)
        if improved:
            atomic_save({"model": model.state_dict(), "config": model_cfg, "classes": CLASSES, "epoch": epoch,
                         "val_dice": dice, "experiment": cfg}, best_p)
        atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "stale": stale,
                     "rng_cpu": torch.get_rng_state(), "rng_sampler": g.get_state()}, last_p)
        row = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5), "val_loss": round(val_loss, 5),
               "val_dice": round(dice, 5), "lr": lr, "seconds": round(time.time() - t0, 1)}
        new = not hist_p.exists()
        with hist_p.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"epoch {epoch + 1}/{epochs} loss {row['train_loss']:.4f} val_loss {val_loss:.4f} "
              f"val_dice {dice:.4f}{' *' if improved else ''} ({row['seconds']}s)", flush=True)

    if args.no_eval:
        return
    from glass_crack_models.eval import run_eval
    ev = ["--checkpoint", str(best_p), "--exp-id", cfg["exp_id"], "--seed", str(seed),
          "--task", f"{cfg['arch']}-{cfg['encoder']}-{cfg.get('loss', 'bce_dice')}",
          "--pretrain", cfg.get("pretrain", ""), "--weighting", cfg.get("weighting", ""),
          "--compute-units", args.compute_units, "--notes", cfg.get("notes", ""), "--device", device]
    if args.smoke:
        ev += ["--limit", "12", "--bootstrap", "200", "--out", str(paths.out_root() / "evals" / name)]
    run_eval.main(ev)


if __name__ == "__main__":
    main()
