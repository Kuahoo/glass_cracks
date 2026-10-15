#!/usr/bin/env python3
"""Convert run_eval cache (.npz of uint8 score maps) into <stem>.npy float maps for
check_ncugd_pipeline.py --backend pred-dir.

  python scripts/npz_to_predmaps.py --list  ~/glass_crack/outputs/evals/E1_s0/cache      # inspect first
  python scripts/npz_to_predmaps.py ~/glass_crack/outputs/evals/E1_s0/cache/<ncugd>.npz  --out outputs/preds_E1_s0_ncugd

Keys are turned into file names with Path(key).stem (so 'ncugd/img01.png' -> img01.npy).
uint8 maps are divided by 255 (uint16 by 65535); float maps are kept.
"""
import argparse, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("path", type=Path, help=".npz file, or cache folder with --list")
ap.add_argument("--list", action="store_true", help="show the .npz files and a few keys/shapes/dtypes, then exit")
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()

if a.list or a.path.is_dir():
    files = sorted(a.path.glob("*.npz")) if a.path.is_dir() else [a.path]
    for f in files:
        with np.load(f) as z:
            ks = list(z.files)
            print(f"{f.name}: {len(ks)} arrays")
            for k in ks[:3]:
                print(f"    key={k!r} shape={z[k].shape} dtype={z[k].dtype} max={z[k].max()}")
    sys.exit(0)

out = a.out or Path("preds_" + a.path.stem)
out.mkdir(parents=True, exist_ok=True)
n = 0
with np.load(a.path) as z:
    for k in z.files:
        m = z[k]
        scale = 255.0 if m.dtype == np.uint8 else 65535.0 if m.dtype == np.uint16 else 1.0
        np.save(out / f"{Path(k).stem}.npy", np.squeeze(m).astype(np.float32) / scale)
        n += 1
print(f"wrote {n} maps to {out}")
