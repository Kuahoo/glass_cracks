# What to do next (plan v4)

Status: Phase 1 done. NCUGD is dropped. Local `data/` is cleaned (~30 GB). Next work is on **TIDE**, driven by `notebooks/tide_v4.ipynb`.

Do **not** block on labeling the ~400 undecided rows in `outputs/review/gdd2_train.csv`. That review is Phase 2 quality work. E8 only needs a later **100 crack + 100 scratch** crop audit, not clearing that queue.

---

## 0. Laptop (before / while opening TIDE)

1. Commit and push any unpushed local changes so TIDE can `git pull`.
2. Confirm `data/extras.zip` exists (~2.8 GB). Rebuild only if needed:
   ```bash
   bash scripts/build_extras.sh
   ```
3. Upload `data/extras.zip` to TIDE as `~/glass_crack/extras.zip`  
   (or put it on Drive and use `EXTRAS_URL=...` in setup).  
   Needed for **E4b / E5 / E6**, not for M1 or E4.

---

## 1. TIDE session

Spawn: **PyTorch Notebook**, 1× L40, 8 CPU, 32 GB RAM.  
Open `notebooks/tide_v4.ipynb` under `~/glass_crack/cracked/`.

### Step A — Pull + setup
Run cell 1 (`git pull`, `scripts/tide/setup.sh`). Confirm `~/glass_crack/bundle/extras/` appears after extras upload/unzip.

### Step B — Archive old E4 (one-time)
Run cell 2 so the old scale/colour-only E4 becomes `E4old`. Without this, the new E4 looks “already finished.”

### Step C — M1 (do this first)
Resolution ablation on frozen `E1_s0` (GPU eval only, ~5–15 min per variant):
- `R0` whole @640  
- `R1` tiles, long side 1280  
- `R2_1024` / `R2_1280` whole  

Then open `$OUT_ROOT/results/m1/compare_runs.md`.  
Decision: whether later training should prefer tiling / larger `imgsz`.

### Step D — Track B: U-Net vs YOLO (architecture compare)
E3 is U-Net ResNet34. Launch with **`train_semseg`**, not `train_yolo` (that caused `KeyError: 'dataset'`).

```bash
CFG=e3_unet_r34.yaml; NAME=E3; SEED=0   # then 1, 2
MODULE=glass_crack_models.train.train_semseg
bash "$CODE_DIR/scripts/tide/job.sh" start ${NAME}_s$SEED $MODULE \
  --config $CFG --seed $SEED --device cuda --workers "$WORKERS" \
  --compute-units "TIDE $GPU_NAME"
```

`E3_s0` already exists; finish seeds 1–2 for a fair 3-seed compare with E1. Then:

```bash
"$PY" -m glass_crack_models.eval.compare_runs --runs E1_s0 E1_s1 E1_s2 E3_s0 E3_s1 E3_s2 \
  --out "$OUT_ROOT/results/yolo_vs_unet"
```

### Step E — M2: E4
Train `e4_yolo_robust.yaml` as `E4`, seeds **0, 1, 2** (one job at a time via `job.sh`, ~2.5 h each).  
Each YOLO recipe must beat E1 on the **threshold-free** compare before you treat the next step as a win.

### Step F — M3: E4b
Requires extras. Train `e4b_yolo_hardneg.yaml` as `E4b`, seeds 0–2.  
Check FP/100 vs E1 without losing recall at P ≥ 0.85.

### Step G — M4: proxies (only after E4/E4b are scored)
1. `E5pre` (`e5_pre_cubit.yaml`, seed 0) → `E5` fine-tune (`e5_yolo_cubit_ft.yaml`, seeds 0–2).  
2. `E6` (CarDD) **only if E5 does not help**.

### Step H — Compare
Run the notebook’s threshold-free comparison / YOLO vs U-Net table for all finished runs.

---

## 2. Parallel (laptop, optional, not a gate)

| Task | When |
|---|---|
| 100 crack + 100 scratch crop audit → E8 | After M1/M2 are rolling; before claiming label quality fixed |
| Finish `gdd2_train` review (~400 empty decisions) | Phase 2; never blocks TIDE M1–M4 |
| Curtain-wall micro test / V1 AGDD dual-view filter | After M-series, or when you have time |

---

## 3. Rules of thumb

- One GPU job at a time; resume by re-running the same `job.sh start …` command.
- Keep the browser/session alive or accept that the **server** dies after 12 h with no connection (background jobs do not survive a full stop).
- Do not train on AGDD (held-out). Do not bring NCUGD back.
- Gate each step on beating E1 threshold-free before stacking the next change.

Primary references: `notebooks/tide_v4.ipynb`, `README.md` §9, `docs/model_plan_V4.md`.
