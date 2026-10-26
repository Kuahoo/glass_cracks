# Glass crack segmentation (SDSU, elicc curtain-glass project)

Status as of 2026-10-15. Research use only; licensing and permissions are deferred on purpose.

**Working approach: phases.** Phase 1 finishes the previous work and fixes open bugs and measurement problems. Only then do Phases 2 to 5 (labels, data and training changes, new model tracks, new data). See Section 9.

- **GitHub repo:** https://github.com/Kuahoo/glass_cracks (remote `origin`)
- **Laptop workspace:** `/home/kuahoo/work/sdsu/2026_fall_sdsu/cracked`
- **TIDE workspace:** `/home/jovyan/glass_crack/` (SDSU JupyterHub)
- **Plan of record:** `model_implementation_plan.md` (phases, experiments E1-E7, evaluation protocol); package pins in `requirements.md`; data plan in `data_preprocessing_plan.md`.

---

## 1. Goal

Segment visible cracks and scratches (merged into one `linear_defect` target) on architectural / curtain-wall glass, working with the elicc curtain glass company. Target deployment is drone imagery of facades. Two model tracks:

- **Track A:** YOLO-seg (`yolo26s-seg`, Ultralytics 8.4.177). E1 = 4 classes (bubble, crack, inclusion, scratch; scored as crack OR scratch). E2 = `linear_defect` only.
- **Track B:** semantic segmentation (`segmentation-models-pytorch`). E3 = U-Net ResNet34, BCE + Dice (equal weight; see Phase 1 note below).
- **Candidate Track C (not started):** SegFormer, then a LoRA-adapted SAM (see Section 8).

---

## 2. Where everything lives

### 2.1 Accounts and links
- GitHub: https://github.com/Kuahoo/glass_cracks.
- SDSU TIDE (JupyterHub): https://jupyterhub-research.sdsu.edu, image **PyTorch Notebook** (torch 2.4.0+cu121, Python 3.11), 1x NVIDIA L40, 8 CPU, 32 GB RAM. Free GPUs: https://nrp.ai/viz/resources/ (filter `rci-tide`). Docs: https://csu-tide.github.io/ (batch jobs: https://csu-tide.github.io/batch-jobs/).
  - Persistent storage is only `/home/jovyan` (75 GiB); pip/conda/apt in the default env revert on restart.
  - The server stops after **12 h with no browser connected**; background jobs survive closed tabs.
  - Kubernetes batch Jobs have no runtime limit and auto-restart (Pods 6 h, Deployments 2 weeks). Use later for CUBIT pretraining and long sweeps.
- Colab (Stage A): finished. Its E1-E3 rows used the old matching, so do not mix them with TIDE rows. After a Colab runtime restart `OUT_ROOT`, `DATA_ROOT` and the working directory are lost.
- SDSU GPU cluster / DGX: access requested, answer pending. Home GTX 1080: debugging only.

### 2.2 Laptop (CPU only; run Python as `env .venv/bin/python ...`)
Repo root `/home/kuahoo/work/sdsu/2026_fall_sdsu/cracked`:

| Path | What it is |
|---|---|
| `glass_crack_models/` | Package: `paths.py`, `protocol.py`, `data/`, `train/`, `eval/` (`run_eval.py`, `match.py`, `metrics.py`, `agdd.py`, `bootstrap.py`, **`compare_runs.py`**, **`agdd_box_scores.py`**), `infer/`, `viz/`, `configs/` (E1-E4), `tests/` (33 unit tests) |
| `notebooks/` | `colab_setup.ipynb`, `tide_setup.ipynb`, **`tide_phase1.ipynb`** (Phase 1 TIDE session) |
| `scripts/tide/` | `env.sh`, `setup.sh` (validates the zip before unzipping), `job.sh` |
| `scripts/` | `profile_datasets.py`, `check_ncugd_pipeline.py`, `npz_to_predmaps.py`, `signedDiff.py` |
| `phase_0/` | `prepare_dataset_bundle.py`, `RESEARCH.md` |
| `docs/profile_summary.md` | Snapshot of the 2026-10-15 dataset profile (full JSON under `outputs/profile/`, git-ignored) |
| `temp/` (git-ignored) | `bundle_colab.zip` (193,309,489 bytes, MD5 `6410cf98724221591a7d8c68ca1ff453`), `glass_crack_dataset_bundle/` (laptop `DATA_ROOT`) |
| `data/` (git-ignored) | raw downloads and `processed_bundle_v1/` |
| `outputs/` (git-ignored) | laptop smoke runs and `profile/` |

Git-ignored: `data/`, `outputs/`, `temp/`, `*.pt`, `*.pth`. Never commit datasets or weights. Never write or commit the Roboflow key (`ROBOFLOW_API_KEY` lives only in the shell environment).

### 2.3 TIDE server (`/home/jovyan/glass_crack/`)

| Path | Contents |
|---|---|
| `cracked/` | git clone; run scripts from here |
| `bundle/` | unzipped `bundle_colab.zip` = `DATA_ROOT` |
| `venv/` | project venv (`--system-site-packages`). Use `~/glass_crack/venv/bin/python`. The default Python lacks `cv2` |
| `cache/` | torch / HF / Ultralytics caches |
| `outputs/` | `OUT_ROOT`: `runs/`, `evals/`, `logs/`, `results/results.csv` |
| `bundle_colab.zip` | the uploaded data bundle |

Eval outputs: `~/glass_crack/outputs/evals/<run>/` for `E1_s0..2`, `E2_s0..2`, `E3_s0` (plus smoke runs). Each has `cache/{valid,test,ncugd,agdd}.npz`, `operating_point.json`, `pr_curves.png`, `summary.json` (full sweeps). NCUGD path: `processed/ncugd_external_test`.

Start and monitor a run:
```sh
source ~/glass_crack/cracked/scripts/tide/env.sh
bash "$CODE_DIR/scripts/tide/job.sh" start E1_s0 glass_crack_models.train.train_yolo \
    --config e1_yolo_4class.yaml --seed 0 --device 0 --workers "$WORKERS" --compute-units "TIDE $GPU_NAME"
bash "$CODE_DIR/scripts/tide/job.sh" status          # tail <name> [lines] | stop <name>
```
Use a new job name per seed, one job at a time. Rerunning an interrupted run resumes. **Re-evaluating a finished run now replaces its results.csv row** (same `exp_id` + `seed` + `checkpoint_sha256`) instead of appending a duplicate. After pushing new code, run notebook cells 1-2 (clone/pull, `setup.sh`). Re-score from cache: `run_eval --checkpoint ... --exp-id E2 --seed 0 --reuse-cache`.

---

## 3. Data

### 3.1 Training data in `bundle_colab.zip`
`processed/glass_merged_seg` (4 classes: 0 bubble, 1 crack, 2 inclusion, 3 scratch; 640x640; train 980 / valid 148 / test 148), `processed/glass_merged_linear`, `processed/ncugd_external_test`, `datasets/agdd_aircraft_glass/data`.

### 3.2 External test sets (never trained on, never used for tuning)
- **NCUGD:** 19 lab images (640x480), 32 crack polygons, class 1 = crack.
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
| ncugd_* | 26 / 19 | Held-out test | Stroke width ~9.5 px (wide labels for 1-2 px hairlines) |
| processed_bundle_v1 | 8,255 | Phase_0 bundle (not yet used by training) | Group-aware split; 13 degenerate polygons; AGDD excluded |

### 3.4 `phase_0/prepare_dataset_bundle.py` (rewritten, working)
Builds `data/processed_bundle_v1/` (YOLO-seg, single class `0: crack`). AGDD excluded by default. Last build: 7,022 train / 1,207 val / 26 NCUGD external. **Not yet used by the training code** (which uses `bundle_colab.zip`).

---

## 4. Evaluation protocol and the matching fix

Fixed constants live in `protocol.py`. Threshold is frozen per run at max F1 on valid. Match: IoU >= 0.3, or overlap after 3 px dilation; strict IoU >= 0.5 also reported; cluster bootstrap; 3 seeds for top configs.

**Matching fix.** One prediction may claim several touching GTs when its IoU with their union meets the threshold. A perfect prediction scores 1.000 on NCUGD and merged valid, 0.994 on merged test (2 GTs below `MIN_REGION_PX = 24`).

---

## 5. Results

### 5.1 TIDE (NVIDIA L40, new matching; mean +/- SD over seeds 0-2; E3 seed 0 only)

| | Test recall | Precision | FP / 100 images | Mask IoU | AGDD pair recall (legacy) | Thresholds |
|---|---|---|---|---|---|---|
| **E1** YOLO 4-class | 0.767 +/- 0.041 | **0.865 +/- 0.008** | **25.0 +/- 1.2** | 0.603 +/- 0.028 | 0.581 +/- 0.107 | 0.30, 0.50, 0.40 |
| **E2** YOLO linear_defect | 0.772 +/- 0.051 | 0.834 +/- 0.032 | 32.9 +/- 7.9 | 0.587 +/- 0.014 | 0.507 +/- 0.051 | 0.35, 0.45, 0.45 |
| E3 U-Net R34 | 0.634 | 0.643 | 74.3 | 0.448 | 0.833 | 0.80 |

Per-run recall: E1 s0/s1/s2 0.811 / 0.730 / 0.760; E2 0.823 / 0.721 / 0.772. NCUGD recall is 0.0 for all.

- E1 is the Track A winner on false positives and precision stability; recall ties E2.
- Seed spread (0.72 to 0.82) dwarfs the E1/E2 gap; the frozen threshold (0.30 to 0.50) drives much of it.
- **Replace the AGDD pair column** with per-view numbers from `tide_phase1.ipynb` (Section 9 step 4) before quoting AGDD again. E3's 0.833 is likely inflated by box scoring of blobby masks.
- E3 loss is `bce + dice` at equal weight (`semseg_common.py`). Tiny positive area may need a higher BCE weight or focal term (try as an ablation after Phase 1).

### 5.2 NCUGD: real out-of-domain failure (not a bug)
Labels, loading and matching verified. Cracks are faint 1-2 px hairlines; long dark lines are unlabeled. Cached E1 maps are all zero. Remaining check (Phase 1 on TIDE): `check_ncugd_pipeline.py --backend model` at 640/1024/1280, to separate "blind" from "below threshold". Report as 0/32 either way.

### 5.3 AGDD `image/` vs `images/` gap: model robustness, not a data bug
Ruled out on TIDE: geometry, global lighting, visibility, polarity. Labels are valid on both views; the model responds less to the `image/` illumination. Remaining cheap check: in-box max scores + crop montages (`agdd_box_scores.py`, already in the Phase 1 notebook).

---

## 6. Scripts

```sh
# Profile datasets (laptop)
env .venv/bin/python scripts/profile_datasets.py --children data temp/glass_crack_dataset_bundle/processed --out outputs/profile

# Threshold-free comparison / AGDD in-box (TIDE, CPU; needs existing evals/)
source ~/glass_crack/cracked/scripts/tide/env.sh && cd "$CODE_DIR"
"$PY" -m glass_crack_models.eval.compare_runs --runs E1_s0 E1_s1 E1_s2 E2_s0 E2_s1 E2_s2 E3_s0
"$PY" -m glass_crack_models.eval.agdd_box_scores --runs E1_s0 E1_s1 E1_s2 E2_s0 E2_s1 E2_s2 E3_s0 --montage E1_s0 E2_s0 E3_s0

# NCUGD with a real model (TIDE, GPU)
"$PY" scripts/check_ncugd_pipeline.py --data "$DATA_ROOT/processed/ncugd_external_test" \
  --backend model --weights "$OUT_ROOT/runs/E1_s0/weights/best.pt" \
  --op-threshold 0.30 --device cuda:0 --out "$OUT_ROOT/ncugd_check/E1_s0"
```

Or run `notebooks/tide_phase1.ipynb` top to bottom.

---

## 7. Pre-processing plan (summary)

Full text: `data_preprocessing_plan.md`. Dataset roles, stroke-width normalization, CUBIT tiling, hard negatives, per-source ablations P0-P7 under group 5-fold CV. Day-1 items (AGDD, threshold-free, profiling) are Phase 1; label audit onward is Phase 2+.

---

## 8. Literature and industry scan (2026-10-14)

Glass-specific work is scarce. Foundation-model adaptation (CrackSAM, Segment Any Crack) helps mainly under blur and domain shift. A 2025 review found zero-shot models miss fine cracks or fire on textures; combined-dataset training did not beat per-dataset training. Commercial facade products (H3 Dynamics, Facadevision AI) publish no accuracy numbers. Implications: keep per-source ablations; add SegFormer then LoRA-SAM as Track C; add motion-blur augmentation; try clDice as an ablation; invest in label quality.

---

## 9. Roadmap (phased)

### Phase 1: finish previous work and fix bugs (in progress)

| # | Task | Status |
|---|---|---|
| 1 | Commit Phase 1 code + docs; push; pull on TIDE | **Do now** (this commit) |
| 2 | Base E4 on E1 (`e4_yolo_robust.yaml`) | **Done** (config resolves to `dataset: merged`) |
| 3 | Threshold-free E1 vs E2 (`compare_runs.py`) | **Code ready**; run on TIDE |
| 4 | AGDD per view (`agdd_box_scores.py` + montages) | **Code ready**; run on TIDE |
| 5 | NCUGD real-model check (`check_ncugd_pipeline.py`) | **Code ready**; run on TIDE (GPU) |
| 6 | Profile every dataset | **Done** (`docs/profile_summary.md`) |
| 7 | Housekeeping: results.csv de-dupe; CUBIT label; E3 loss note; zip validation in `setup.sh` | **Done** |
| — | Start E4_s0..2 on TIDE (background) | After pull; 3 seeds |

**Exit criterion:** E1/E2/E3 compared with threshold-free metrics, AGDD reported per view, NCUGD explained, datasets profiled, git in sync, README/results updated. Then start Phase 2.

Open TIDE notebook: `notebooks/tide_phase1.ipynb`.

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
- Not obtained: GCC-385 and Zhou curtain-wall dataset.
- Crack/scratch labels are inconsistent; valid/test are small (~90 cracks per split).
- Roboflow stretched images to 640; physical crack width cannot be recovered.

## 11. Gotchas

- Colab: after a runtime restart, rerun the setup cells.
- Cursor tools sometimes write CRLF; `.gitattributes` forces LF for `*.sh`; check with `bash -n` and `file scripts/*`.
- Laptop: use `env .venv/bin/python ...`.
- TIDE: use `~/glass_crack/venv/bin/python`. Do not mix Colab rows (old matching) with TIDE rows.
