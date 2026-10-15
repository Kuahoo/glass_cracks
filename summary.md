# Glass crack detection: project summary

Status as of 2026-10-14. Hand-off notes for continuing the work.

- **GitHub repo:** https://github.com/Kuahoo/glass_cracks (remote `origin`)
- **Local workspace:** `/home/kuahoo/work/sdsu/2026_fall_sdsu/cracked`
- **Plan of record:** `model_implementation_plan.md` (phases, experiments E1-E7, evaluation protocol). Package pins: `requirements.md`.

## 1. Goal

Segment visible cracks (and scratches, as one `linear_defect` target) on architectural / facade glass. Two model tracks:

- **Track A:** YOLO-seg (`yolo26s-seg`, Ultralytics 8.4.177). E1 = 4 classes (bubble, crack, inclusion, scratch; scored as crack OR scratch), E2 = `linear_defect` only.
- **Track B:** semantic segmentation (`segmentation-models-pytorch`). E3 = U-Net ResNet34, BCE + Dice.

## 2. Repository layout

| Path | What it is |
|---|---|
| `glass_crack_models/` | The package: `paths.py` (all paths from `DATA_ROOT` / `OUT_ROOT`), `protocol.py` (fixed eval constants), `data/` (relabel, rasterize, CUBIT tiling), `train/` (`train_yolo.py`, `train_semseg.py`, resumable), `eval/` (`run_eval.py`, `match.py`, `metrics.py`, `agdd.py`, `bootstrap.py`), `infer/` (tiled multi-scale `predict.py`), `viz/`, `configs/` (E1-E4 YAML), `tests/` (unit tests) |
| `notebooks/colab_setup.ipynb` | Stage A (Colab) session notebook |
| `notebooks/tide_setup.ipynb` | Stage B (TIDE) session notebook: setup, background training jobs, monitor, re-score, results table |
| `scripts/tide/` | `env.sh` (paths + per-session worker sizing), `setup.sh` (idempotent venv / data / relabel), `job.sh` (`start` / `status` / `tail` / `stop` background jobs) |
| `phase_0/prepare_dataset_bundle.py` | Separate multi-dataset YOLO-seg bundle builder (single class `crack`), see Section 4 |
| `phase_0/RESEARCH.md` | Background research notes |
| `temp/` (git-ignored) | `bundle_colab.zip` (training data bundle, 193,309,489 bytes, MD5 `6410cf98724221591a7d8c68ca1ff453`), `clean_datasets.py` (built `processed/`), `glass_crack_dataset_bundle/` (laptop `DATA_ROOT`) |
| `data/` (git-ignored) | Raw downloaded datasets and `processed_bundle_v1/` (output of the phase_0 script) |
| `outputs/` (git-ignored) | Laptop smoke runs |

## 3. Training data used by the models (`bundle_colab.zip`)

`processed/glass_merged_seg` (4 classes), `processed/glass_merged_linear` (built on each machine by `python -m glass_crack_models.data.relabel_classes`), `processed/ncugd_external_test`, and `datasets/agdd_aircraft_glass/data`. Valid and test are 148 images each.

External test sets (never trained on, never used for tuning):
- **NCUGD:** 19 lab images, 32 crack polygons (class 1 = crack).
- **AGDD:** aircraft glass, box labels (classes 0 contusion, 1 scratches, 2 crack, 3 spot), two image views `images/` and `image/` sharing one label set. Scored at box level; "pair recall" = found in either view.

## 4. `phase_0/prepare_dataset_bundle.py` (rewritten, working)

Builds `data/processed_bundle_v1/` (YOLO-seg, single class `0: crack`). Run from the repo root:

```sh
.venv/bin/python phase_0/prepare_dataset_bundle.py --data-root data --out-dir data/processed_bundle_v1
.venv/bin/python phase_0/prepare_dataset_bundle.py --limit 20 --out-dir temp/bundle_smoke   # quick smoke run
```

Bugs fixed: SSGD crash on flat segmentations; crackseg9k base64 PNGs silently produced 0 samples; crackSegUltralytics label path never matched (all images became negatives); AGDD label path wrong and all classes mapped to crack; GDD2 bubbles/inclusions labeled as cracks; "clean negatives" contained cracks; thin cracks dropped by `contourArea`; random per-tile split leaked overlapping tiles and Roboflow copies across train/val.

Now: per-dataset class filters (scratch + crack kept), crack holes preserved, truly clean negatives (empty-label GSEG images, defect-free CUBIT tiles capped at 600, empty-mask FCD), deterministic group-aware split, `manifest.csv`, `summary_report.json`, safe overwrite of a previous bundle. **AGDD is excluded by default** (`--exclude` defaults to `agdd`) so it stays an external test. Last build: 7,022 train / 1,207 val / 26 NCUGD external; Ultralytics loads it with 0 corrupt labels; 0 groups leak across splits. This bundle is **not yet used** by the training code (that uses `bundle_colab.zip`).

## 5. Evaluation changes and findings

**Matching fix (`eval/match.py`, `eval/metrics.py`).** Old greedy one-to-one matching capped a *perfect* prediction at 0.719 recall on NCUGD (touching GT cracks merge into one predicted region). Now one prediction may claim several touching GTs when its IoU with their union meets the threshold (grown greedily, ties admit nested GTs; a big blob that merely touches two GTs still gets one). Recall counts matched GTs; precision counts matched predictions (`tp_pred`). Perfect prediction now scores 1.000 on NCUGD and merged valid, 0.994 on merged test (2 GTs of 17 and 23 px are below `MIN_REGION_PX = 24` and can never match). 28 unit tests pass. Fragment predictions of one GT still count as extra FPs (unchanged). Documented in the plan.

**NCUGD recall = 0 is real, not a bug.** Labels, image loading and matching were verified. NCUGD cracks are faint 1-2 px hairlines starting at bright dots on grayscale lab glass; the long dark lines across those images are unlabeled. Cached score maps show max score within ~7 px of any crack is 0.00-0.03 for all models at every threshold; E2/E3 fire confidently elsewhere (max 0.64 / 0.96). Report NCUGD as an out-of-domain failure (0/32).

**AGDD view gap (open).** Recall on the `image/` view is far below `images/` (Colab: about 0.05-0.13 vs 0.50-0.85) with the same labels. Not yet investigated; likely a data/geometry mismatch between views.

## 6. Compute

- **Laptop:** CPU only, `.venv` (Python 3.13). Run Python as `env .venv/bin/python ...` (the bare path can fail inside the Cursor sandbox).
- **Colab (Stage A, done):** first E1-E3 runs on a T4. Those rows used the old matching; do not mix them with TIDE rows.
- **SDSU TIDE (Stage B, active):** https://jupyterhub-research.sdsu.edu, image **PyTorch Notebook** (torch 2.4.0+cu121, Python 3.11), 1 GPU (NVIDIA L40), 8 CPU, 32 GB RAM. Free GPUs: https://nrp.ai/viz/resources/ (filter `rci-tide`). Docs: https://csu-tide.github.io/.
  - Persistent storage is only `/home/jovyan` (75 GiB); everything else (pip/conda/apt in the default env) reverts on restart.
  - The server stops after **12 hours with no browser connected**. Background jobs survive closed tabs; reconnect within 12 h.
  - Batch jobs (https://csu-tide.github.io/batch-jobs/): Kubernetes Jobs have no runtime limit and auto-restart (Pods 6 h, Deployments 2 weeks). Needs `kubectl`, batch access, a container image and mountable storage. Use later for CUBIT pretraining / long sweeps.

**TIDE layout** (`~/glass_crack/`): `cracked/` (git clone), `bundle/` (unzipped data, `DATA_ROOT`), `venv/` (`--system-site-packages`, pins installed), `cache/` (torch / HF / Ultralytics), `outputs/` (`OUT_ROOT`: `runs/`, `evals/`, `logs/`, `results/results.csv`), `bundle_colab.zip`.

**TIDE session:** notebook cells 1-2 only after pushing new code (clone/pull, `setup.sh`). Start a run:

```sh
source ~/glass_crack/cracked/scripts/tide/env.sh
bash "$CODE_DIR/scripts/tide/job.sh" start E1_s0 glass_crack_models.train.train_yolo \
    --config e1_yolo_4class.yaml --seed 0 --device 0 --workers "$WORKERS" --compute-units "TIDE $GPU_NAME"
bash "$CODE_DIR/scripts/tide/job.sh" status          # tail <name> [lines] | stop <name>
```

Use a new job name per seed; run one job at a time; rerunning an interrupted run resumes; rerunning a finished run re-evaluates and appends a duplicate results row. Re-score from cached predictions: `run_eval --checkpoint ... --exp-id E2 --seed 0 --reuse-cache`.

## 7. Results (TIDE, NVIDIA L40, new matching)

Mean ± SD over seeds 0-2 (E3 seed 0 only). Threshold is frozen per run at max F1 on valid.

| | Test recall | Precision | FP / 100 images | Mask IoU | AGDD pair recall | Thresholds |
|---|---|---|---|---|---|---|
| **E1** YOLO 4-class | 0.767 ± 0.041 | **0.865 ± 0.008** | **25.0 ± 1.2** | 0.603 ± 0.028 | 0.581 ± 0.107 | 0.30, 0.50, 0.40 |
| **E2** YOLO linear_defect | 0.772 ± 0.051 | 0.834 ± 0.032 | 32.9 ± 7.9 | 0.587 ± 0.014 | 0.507 ± 0.051 | 0.35, 0.45, 0.45 |
| E3 U-Net R34 | 0.634 | 0.643 | 74.3 | 0.448 | 0.833 | 0.80 |

Per run: E1 s0/s1/s2 recall 0.811 / 0.730 / 0.760; E2 0.823 / 0.721 / 0.772. NCUGD recall 0.0 for all.

Conclusions:
- **E1 is the Track A winner:** recall tied with E2, but fewer false positives in every seed and much more stable precision.
- Seed-to-seed spread (0.72-0.82 recall) dwarfs the E1/E2 recall gap; single-seed comparisons are unreliable. The frozen threshold swings 0.30-0.50 (148 valid images), which drives much of the recall variance.
- Versus Colab (about 0.73 recall), the matching fix plus retraining added about 4 points on average.
- E3 trails on test; its high AGDD number is likely inflated by box-level scoring of blobby masks.

## 8. Next steps (in recommended order)

1. **Sync git first.** `origin/main` has a newer commit (`576eacc cracked files`) that the laptop does not have, and the laptop has an uncommitted change to `scripts/tide/setup.sh` (validates `bundle_colab.zip` before unzipping and explains HTML / truncated uploads). Run `git pull`, resolve any `.gitignore` conflict, commit, push.
2. **Base E4 on E1.** `glass_crack_models/configs/e4_yolo_robust.yaml` currently has `base: e2_yolo_linear.yaml`; switch it to the E1 (4-class) setup before running the robustness experiment.
3. **Threshold-free comparison.** Each `evals/<run>/summary.json` holds full precision/recall sweeps; compare E1 vs E2 by recall at fixed precision (e.g. 0.85) or PR-curve area, averaged over seeds. No GPU needed. Consider a larger validation set to stabilize the frozen threshold.
4. **Investigate the AGDD `image/` vs `images/` gap** so the external number can be trusted.
5. Later: decide whether to train on `data/processed_bundle_v1` (phase_0 bundle), CUBIT pretraining via TIDE batch Jobs, NCUGD false-positive overlays (what E2/E3 fire on), optionally upscaled inference for hairline cracks.

## 9. Gotchas

- Colab: after a runtime restart, `OUT_ROOT` / `DATA_ROOT` and the working directory are lost; rerun the setup cells (this caused the earlier `zip error: Nothing to do` and `No module named 'glass_crack_models'`).
- Files written by the Cursor tools sometimes get CRLF line endings; bash rejects CRLF scripts. `.gitattributes` forces LF for `*.sh`; check new scripts with `bash -n`.
- `.gitignore` excludes `data/`, `outputs/`, `temp/`, `*.pt`, `*.pth`: never commit datasets or weights.
- Never write or commit the Roboflow API key (`ROBOFLOW_API_KEY` lives only in the shell environment).
- Licenses are deferred (plan Section 9); Ultralytics is AGPL-3.0.
