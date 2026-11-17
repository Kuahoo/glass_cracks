# Glass crack segmentation (SDSU, elicc curtain-glass project)

Status as of 2026-11-05 (plan v4). Research use only; licensing and permissions are deferred on purpose.

**Working approach:** Phase 1 (measurements, bug fixes) is done. Plan v4 (`agent_plan.md`, Section 4): M1/E4/E4b/E5pre/E6pre are scored on TIDE (E4/E4b did **not** beat E1 threshold-free). Remaining: Track C RF-DETR (E7), E3_s1 eval, optional E5/E6 glass fine-tunes, E8 label audit. Each **YOLO** recipe must beat E1 before stacking. Tracks B/C are architecture compares. See Section 9 and `NEXT.md`.

- **GitHub repo:** https://github.com/Kuahoo/glass_cracks (remote `origin`)
- **Laptop workspace:** `/home/kuahoo/work/sdsu/2026_fall_sdsu/cracked`
- **TIDE workspace:** `/home/jovyan/glass_crack/` (SDSU JupyterHub)
- **Plan of record:** `agent_plan.md` (v4: datasets, work order M0-M4, register E1/R0-R2/E4/E4b/E8/E5/E6); older `model_implementation_plan.md` for the evaluation protocol; package pins in `requirements.md`; data plan in `data_preprocessing_plan.md`.

---

## 1. Goal

Segment visible cracks and scratches (merged into one `linear_defect` target) on architectural / curtain-wall glass, working with the elicc curtain glass company. Target deployment is drone imagery of facades. Three model tracks:

- **Track A:** YOLO-seg (`yolo26s-seg`, Ultralytics 8.4.177). E1 = 4 classes (bubble, crack, inclusion, scratch; scored as crack OR scratch). E2 = `linear_defect` only.
- **Track B:** semantic segmentation (`segmentation-models-pytorch`, kept for academic comparisons against Track A). Config keys `arch` (unet, unetplusplus, fpn, deeplabv3plus, segformer, ...) and `encoder` (resnet34, resnet50, mit_b2, ...); defaults `unet` + `resnet34`. E3 = U-Net ResNet34, BCE + Dice (equal weight).
- **Track C:** RF-DETR-Seg (`e7_rfdetr_seg.yaml`, default Large). Trained via `train_rfdetr.py`; scored with the same `run_eval` protocol for architecture compare vs E1/E3.
- **Parked:** LoRA-adapted SAM / SegFormer (see Section 8; former “Track C” ideas now sit behind RF-DETR).

---

## 2. Where everything lives

### 2.1 Accounts and links
- GitHub: https://github.com/Kuahoo/glass_cracks.
- SDSU TIDE (JupyterHub): https://jupyterhub-research.sdsu.edu, image **PyTorch Notebook** (torch 2.4.0+cu121, Python 3.11), 1x NVIDIA L40, 8 CPU, 32 GB RAM. Free GPUs: https://nrp.ai/viz/resources/ (filter `rci-tide`). Docs: https://csu-tide.github.io/ (batch jobs: https://csu-tide.github.io/batch-jobs/).
  - Persistent storage is only `/home/jovyan` (75 GiB); pip/conda/apt in the default env revert on restart.
  - The server stops after **12 h with no browser connected**; background jobs survive closed tabs.
  - Kubernetes batch Jobs have no runtime limit and auto-restart (Pods 6 h, Deployments 2 weeks). Use later for CUBIT pretraining and long sweeps.
- Colab (Stage A): finished. Its E1-E3 rows used the old matching, so do not mix them with TIDE rows.
- SDSU GPU cluster / DGX: access requested, answer pending. Home GTX 1080: debugging only.

### 2.2 Laptop (CPU only; run Python as `env .venv/bin/python ...`)
Repo root `/home/kuahoo/work/sdsu/2026_fall_sdsu/cracked`:

| Path | What it is |
|---|---|
| `glass_crack_models/` | Package: `paths.py`, `protocol.py`, `data/`, `train/` (`train_yolo.py` Track A, `train_semseg.py` Track B, `train_rfdetr.py` Track C), `eval/` (`run_eval.py`, `match.py`, `metrics.py`, `agdd.py`, `bootstrap.py`, **`compare_runs.py`**, **`agdd_box_scores.py`**), `infer/` (YOLO / U-Net / RF-DETR predictors), `viz/`, `configs/` (`e1`–`e6` + `e7_rfdetr_seg`), `tests/` |
| `notebooks/` | `tide_setup.ipynb` (first-time TIDE setup, E1-E3 jobs), **`tide_v4.ipynb`** (current plan v4 session) |
| `scripts/tide/` | `env.sh`, `setup.sh` (validates and unzips `bundle_colab.zip`, then optional `extras.zip`), `job.sh` |
| `scripts/` | **`build_extras.sh`** (laptop: builds `data/extras/` + `data/extras.zip`), `profile_datasets.py` |
| `phase_0/` | `prepare_dataset_bundle.py` (also builds the CarDD and hard-negative extras via `--only`), `RESEARCH.md` |
| `docs/profile_summary.md` | Snapshot of the 2026-10-15 dataset profile (full JSON under `outputs/profile/`, git-ignored) |
| `temp/` (git-ignored) | `bundle_colab.zip` (193,309,489 bytes, MD5 `6410cf98724221591a7d8c68ca1ff453`), `glass_crack_dataset_bundle/` (laptop `DATA_ROOT`) |
| `data/` (git-ignored) | raw downloads (cleaned 2026-11-05, Section 3.5), `rf_*/` Roboflow downloads, `extras/` + `extras.zip` (plan v4 sets, Section 3.4) |
| `outputs/` (git-ignored) | laptop smoke runs and `profile/` |

Git-ignored: `data/`, `outputs/`, `temp/`, `*.pt`, `*.pth`. Never commit datasets or weights. Never write or commit the Roboflow key (`ROBOFLOW_API_KEY` lives only in the shell environment).

### 2.3 TIDE server (`/home/jovyan/glass_crack/`)

| Path | Contents |
|---|---|
| `cracked/` | git clone; run scripts from here |
| `bundle/` | unzipped `bundle_colab.zip` = `DATA_ROOT`; `bundle/extras/` from `extras.zip` (zip deleted after unzip) |
| `venv/` | project venv (`--system-site-packages`). Use `~/glass_crack/venv/bin/python`. The default Python lacks `cv2` |
| `cache/` | torch / HF / Ultralytics caches |
| `outputs/` | `OUT_ROOT`: `runs/`, `evals/`, `logs/`, `results/results.csv` |
| `bundle_colab.zip` | the uploaded data bundle |

Eval outputs: `~/glass_crack/outputs/evals/<run>/` for `E1_s0..2`, `E2_s0..2`, `E3_s0` (plus smoke runs). Each has `cache/{valid,test,agdd}.npz`, `operating_point.json`, `pr_curves.png`, `summary.json` (full sweeps).

Start and monitor a run:
```sh
source ~/glass_crack/cracked/scripts/tide/env.sh
bash "$CODE_DIR/scripts/tide/job.sh" start E1_s0 glass_crack_models.train.train_yolo \
    --config e1_yolo_4class.yaml --seed 0 --device 0 --workers "$WORKERS" --compute-units "TIDE $GPU_NAME"
bash "$CODE_DIR/scripts/tide/job.sh" status          # tail <name> [lines] | stop <name>
```
Use a new job name per seed, one job at a time. Rerunning an interrupted run resumes. **Re-evaluating a finished run now replaces its results.csv row** (same `exp_id` + `seed` + `checkpoint_sha256`) instead of appending a duplicate. After pushing new code, run cell 1 of `tide_v4.ipynb` (pull, `setup.sh`). Re-score from cache: `run_eval --checkpoint ... --exp-id E2 --seed 0 --reuse-cache`.

---

## 3. Data

### 3.1 Training data in `bundle_colab.zip`
`processed/glass_merged_seg` (4 classes: 0 bubble, 1 crack, 2 inclusion, 3 scratch; 640x640; train 980 / valid 148 / test 148), `processed/glass_merged_linear`, `datasets/agdd_aircraft_glass/data` (plus an unused legacy test folder).

### 3.2 External test sets (never trained on, never used for tuning)
- **AGDD:** aircraft glass, box labels, two views `images/` and `image/` sharing one label set; scored at box level. Report **per view**; `images/` is the main external number. Do not average the views.

### 3.3 Other datasets (profiled 2026-10-15; see `docs/profile_summary.md`)
| Dataset | Images | Role | Profile verdict |
|---|---|---|---|
| glass_defect_detection2 / glass_segmentation_2026 | 578 / 1,176 | Train/valid/test core | YOLO polygons, stretched to 640; empty-label rates 27% / 6.5% |
| cubit_inseg | 6,996 | Main crack **pretraining** (drone facade; **not glass**) | YOLO polygons, classes crack / spalling; needs tiling; 8 degenerate polygons |
| fcd_108_* | 281 / 270 | Localization pretraining | PNG masks, stroke width ~4 px @640 |
| SSGD | 2,504 | Pretraining candidate | **COCO polygons** (flat lists), not boxes. Classes include crack, scratch, broken, spot, … Crack stroke ~26-35 px @640 |
| CarDD | 12,000 files (4,000 annotated) | Optional ablation only | COCO + thick PNG masks (stroke ~256 px); low expected value |
| Ultralytics crack-seg | 4,029 | Generic crack pretraining | YOLO polygons, 416x416, stroke ~8 px |
| CrackSeg9k | (parquet) | Generic crack pretraining | Images embedded in `train.parquet` / `test.parquet`; materialize via `phase_0` |
| curtain_wall (Brandon-Johns, CC BY 4.0) | 140 (+60 glass/frame/other masks) | **GEOMETRY_ONLY**: crack-free hard negatives | Cloned 2026-11-05 into `data/curtain_wall`; 2048x1536 PNG; no crack labels |
| GSDNet CGD / AiGD / RSGD | - | **BLOCKED** (Baidu Pan, needs a Baidu account: BaiduPCS-Go + BDUSS) | Links/codes in `agent_plan.md` §1 |
| Roboflow glass-crack-detection | - | **BLOCKED_DELETED**; replaced by Ultralytics crack-seg (already on disk) | |
| Roboflow `links.md` sets (`scripts/fetch_roboflow.py`) | tongji 684, spman 98 (x5 aug), skyvision 78, gdd1296 v1 1,296 | Pane-level broken glass; external curtain-wall test; windshield proxy | Labels mark whole shattered panes (boxes / coarse polygons), not crack lines. glass-hshyy dropped (same images as tongji); bsc off-domain |

### 3.4 Plan v4 extras (`scripts/build_extras.sh` -> `data/extras.zip`, 2,913,859,740 bytes, MD5 `94097683c3cb8069a1d8d8617a4fa452`)
Built 2026-11-05 on the laptop; unzips to `$DATA_ROOT/extras/` on TIDE (upload to `~/glass_crack/extras.zip`, or Google Drive + `EXTRAS_URL=<link> bash scripts/tide/setup.sh`). Generated `data.yaml` files have no `path:` key, so they work wherever they are unzipped.

| Set | Contents | Used by |
|---|---|---|
| `extras/cubit_tiles/` | CUBIT CR1 crack tiles, 1024 px windows (25% overlap) -> 640, class `linear_defect`; 58,844 tiles (51,845 with cracks), CUBIT's own train/valid/test split | E5pre |
| `extras/cardd/` | CarDD `crack` + `glass shatter`, single class; 1,077 train / 183 val images (group-aware split) | E6pre |
| `extras/broken_pane/` | tongji + spman, class `broken_glass`; 333 train / 119 valid / 68 test (171 crack-free facade images). Augmentation copies and 262 near-duplicates removed (tongji repeated crops and mirrors across its own splits); none overlap the glass core or AGDD | Pane-level detector / hard examples (not yet in a config) |
| `extras/curtainwall_test/` | skyvision facade photos, class `broken_glass`, 78 test images | External curtain-wall test (plan §5), scored per pane |
| `extras/windshield/` | gdd1296 v1 (chip, scratch); built once `fetch_roboflow.py gdd1296` has been run | Optional automotive proxy, like CarDD |
| `extras/hardneg/images/train/` | 1,441 crack-free tiles: 840 curtain-wall, 600 clean CUBIT, 1 facade; 740 carry synthetic glare streaks | E4b (30% of the train list) |

### 3.5 Data cleanup
**2026-11-05:** Removed CUBIT `images.zip`/`labels.zip` (exact copies of the extracted folders, 18.7 GB), `CarDD_release/CarDD_SOD` (only `CarDD_COCO` is used), `processed_bundle_v1` (never used for training; rebuild with `phase_0/prepare_dataset_bundle.py`), `ncugd_glass_crack` (dropped), `SSGD/__MACOSX`, `curtain_wall/.git` and its code examples, `rf_hshyy` (same images as tongji) and the augmented gdd1296 v2.

**2026-11-16:** Confirmed NCUGD gone from `data/`; deleted leftover `ncugd_check` and eval/predict NCUGD caches under `outputs/`. Removed parked unused sets `SSGD` (~1.1 GB) and `crackseg9k` (~4 GB), and superseded raw `rf_skyvision` (replaced by `extras/curtainwall_test`). Still on disk as optional contingency: `crackSegUltralytics` (~122 MB). Cubit raw (`cubit_inseg`, ~20 GB) kept so tiles can be rebuilt; `extras.zip` (~2.8 GB) kept for TIDE upload alongside `extras/`.

---

## 4. Evaluation protocol and the matching fix

Fixed constants live in `protocol.py`. Threshold is frozen per run at max F1 on valid. Match: IoU >= 0.3, or overlap after 3 px dilation; strict IoU >= 0.5 also reported; cluster bootstrap; 3 seeds for top configs.

**Matching fix.** One prediction may claim several touching GTs when its IoU with their union meets the threshold. A perfect prediction scores 1.000 on merged valid, 0.994 on merged test (2 GTs below `MIN_REGION_PX = 24`).

---

## 5. Results

### 5.1 TIDE (NVIDIA L40, new matching; mean +/- SD over seeds 0-2; E3 seed 0 only)

| | Test recall | Precision | FP / 100 images | Mask IoU | AGDD pair recall (legacy) | Thresholds |
|---|---|---|---|---|---|---|
| **E1** YOLO 4-class | 0.767 +/- 0.041 | **0.865 +/- 0.008** | **25.0 +/- 1.2** | 0.603 +/- 0.028 | 0.581 +/- 0.107 | 0.30, 0.50, 0.40 |
| **E2** YOLO linear_defect | 0.772 +/- 0.051 | 0.834 +/- 0.032 | 32.9 +/- 7.9 | 0.587 +/- 0.014 | 0.507 +/- 0.051 | 0.35, 0.45, 0.45 |
| E3 U-Net R34 | 0.634 | 0.643 | 74.3 | 0.448 | 0.833 | 0.80 |

Per-run recall: E1 s0/s1/s2 0.811 / 0.730 / 0.760; E2 0.823 / 0.721 / 0.772.

- E1 is the Track A winner on false positives and precision stability; recall ties E2.
- Seed spread (0.72 to 0.82) dwarfs the E1/E2 gap; the frozen threshold (0.30 to 0.50) drives much of it.
- **Replace the AGDD pair column** with per-view numbers (`compare_runs.md`, `agdd_box_scores.md`) before quoting AGDD again. E3's 0.833 is likely inflated by box scoring of blobby masks.
- E3 loss is `bce + dice` at equal weight (`semseg_common.py`). Tiny positive area may need a higher BCE weight or focal term (try as an ablation after Phase 1).

### 5.2 AGDD `image/` vs `images/` gap: model robustness, not a data bug
Ruled out on TIDE: geometry, global lighting, visibility, polarity. Labels are valid on both views; the model responds less to the `image/` illumination. Remaining cheap check: in-box max scores + crop montages (`agdd_box_scores.py`; results in `outputs/results/agdd_box_scores.md`).

---

## 6. Scripts

```sh
# Profile datasets (laptop)
env .venv/bin/python scripts/profile_datasets.py --children data temp/glass_crack_dataset_bundle/processed --out outputs/profile

# Threshold-free comparison / AGDD in-box (TIDE, CPU; needs existing evals/)
source ~/glass_crack/cracked/scripts/tide/env.sh && cd "$CODE_DIR"
"$PY" -m glass_crack_models.eval.compare_runs --runs E1_s0 E1_s1 E1_s2 E2_s0 E2_s1 E2_s2 E3_s0
"$PY" -m glass_crack_models.eval.agdd_box_scores --runs E1_s0 E1_s1 E1_s2 E2_s0 E2_s1 E2_s2 E3_s0 --montage E1_s0 E2_s0 E3_s0

# M1 resolution ablation on a frozen checkpoint (TIDE, GPU); one --exp-id per variant
"$PY" -m glass_crack_models.eval.run_eval --checkpoint "$OUT_ROOT/runs/E1_s0/weights/best.pt" --exp-id R1 --seed 0 \
  --imgsz 1280 --device 0 --no-map          # R0: --imgsz 640 --whole; R2: --imgsz 1024|1280 --whole
# Track B training (any smp arch/encoder via the config; defaults unet + resnet34 = E3). Same job.sh pattern as Track A.
bash "$CODE_DIR/scripts/tide/job.sh" start E3_s0 glass_crack_models.train.train_semseg \
  --config e3_unet_r34.yaml --seed 0 --device 0 --workers "$WORKERS" --compute-units "TIDE $GPU_NAME"
# Same R0-R2 flags work on a Track B checkpoint: --checkpoint "$OUT_ROOT/runs/E3_s0/best.pth"

# Predict on new photos (Track A .pt or Track B .pth)
"$PY" -m glass_crack_models.infer.predict --checkpoint "$OUT_ROOT/runs/E1_s0/weights/best.pt" --input photo.jpg --out preds/
```

Or run `notebooks/tide_v4.ipynb` top to bottom.

---

## 7. Pre-processing plan (summary)

Full text: `data_preprocessing_plan.md`. Dataset roles, stroke-width normalization, CUBIT tiling, hard negatives, per-source ablations P0-P7 under group 5-fold CV. Day-1 items (AGDD, threshold-free, profiling) are Phase 1; label audit onward is Phase 2+.

---

## 8. Literature and industry scan (2026-10-14)

Glass-specific work is scarce. Foundation-model adaptation (CrackSAM, Segment Any Crack) helps mainly under blur and domain shift. A 2025 review found zero-shot models miss fine cracks or fire on textures; combined-dataset training did not beat per-dataset training. Commercial facade products (H3 Dynamics, Facadevision AI) publish no accuracy numbers. Implications: keep per-source ablations; add SegFormer then LoRA-SAM as Track C; add motion-blur augmentation; try clDice as an ablation; invest in label quality.

---

## 9. Roadmap (phased)

### Phase 1: finish previous work and fix bugs (done)
Threshold-free E1/E2/E3 comparison, AGDD per view, dataset profiling, housekeeping. E1 is the base (test recall @P>=0.85 0.825 +/- 0.010, PR area 0.828).
The first E4 run (scale/colour only, seed 0) is archived as `E4old` by `tide_v4.ipynb` cell 2.

### Plan v4 work order (current; `notebooks/tide_v4.ipynb`)

| Step | Run | Config | Status |
|---|---|---|---|
| M0 | Baseline E1/E2/E3, threshold-free + AGDD per view | - | **Done** (`compare_runs.md`, `agdd_box_scores.md`) |
| M1 | R0 / R1 / R2_1024 / R2_1280 on frozen E1_s0 | `run_eval --imgsz/--whole` | **Ready** |
| M2 | E4: E1 + scale, colour, motion blur, JPEG (3 seeds) | `e4_yolo_robust.yaml` | **Ready** |
| M3 | E4b: E4 + 30% hard negatives (3 seeds) | `e4b_yolo_hardneg.yaml` | **Ready** (needs `extras.zip`) |
| M3 | E8: E1 on audited crack/scratch labels | - | Waits for the manual 100 + 100 crop audit |
| M4 A | E5pre (CUBIT, 30 epochs, seed 0) -> E5 fine-tune (3 seeds) | `e5_pre_cubit.yaml`, `e5_yolo_cubit_ft.yaml` | **Ready** (needs `extras.zip`) |
| M4 B | E6pre (CarDD, seed 0) -> E6 fine-tune (3 seeds), only if E5 does not help | `e6_pre_cardd.yaml`, `e6_yolo_cardd_ft.yaml` | **Ready** (needs `extras.zip`) |
| - | V1 AGDD dual-view filter; curtain-wall micro test set (plan §5) | - | Not started |

Config keys added for v4 (see `train_yolo.py` docstring): `data` (proxy data.yaml), `hard_negatives`, `albu` (Albumentations), `eval: false`; `model` may be a checkpoint path; nested `train:` blocks now merge with the base config.

### Phase 2: labels and measurement quality
Crack/scratch audit; FP gallery review; group 5-fold CV.

### Phase 3: data and training improvements
Hard negatives; letterbox + scale; blur/vibration aug; per-source ablations P0-P7; TTA / ensemble.

### Phase 4: new model tracks
SegFormer; LoRA-SAM; clDice; weight averaging.

### Phase 5: new data
elicc drone imagery; multi-view; optional pseudo-labels / CUBIT batch Jobs.

---

## 10. Open items and caveats

- Deferred: licenses (Ultralytics AGPL-3.0; SSGD academic-only; CarDD and AGDD non-commercial; Roboflow upstream rights unverified), elicc agreements, privacy.
- Not obtained: GCC-385, Zhou curtain-wall dataset (authors unresponsive), GSDNet CGD (Baidu account needed).
- Crack/scratch labels are inconsistent; valid/test are small (~90 cracks per split).
- Roboflow stretched images to 640; physical crack width cannot be recovered.

## 11. Gotchas

- Cursor tools sometimes write CRLF; `.gitattributes` forces LF for `*.sh`; check with `bash -n` and `file scripts/*`.
- Laptop: use `env .venv/bin/python ...`.
- TIDE: use `~/glass_crack/venv/bin/python`. Do not mix Colab rows (old matching) with TIDE rows.
