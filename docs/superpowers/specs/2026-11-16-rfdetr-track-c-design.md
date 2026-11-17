# Track C: RF-DETR-Seg architecture compare

**Date:** 2026-11-16  
**Status:** Design approved (Approach 1) — awaiting user review of this spec before implementation plan  
**Branch:** `models`

## Goal

Add a third model track so we can compare **YOLO-seg (Track A / E1)**, **U-Net (Track B / E3)**, and **RF-DETR-Seg (Track C / E7)** under the **same** glass evaluation protocol (`run_eval`, threshold-free recall @ P≥0.85, AGDD per view).

## Decision

**Approach 1 — thin adapter around the official `rfdetr` package.**  
Do not invent a parallel metrics path. Train with RF-DETR’s API; adapt outputs into our existing score-map predictors so `run_eval` / `compare_runs` stay unchanged.

## Scope (this scaffold)

| In | Out |
|---|---|
| `train_rfdetr.py`, E7 config, smoke flag | Hard-neg / CUBIT / CarDD pretrain for RF-DETR |
| `RfDetrPredictor` + `load_predictor` routing | Full 3-seed sweep until smoke passes |
| Hook into post-train `run_eval` | Changing the matching protocol |
| `requirements.md` pin + TIDE/`NEXT.md`/`tide_v4` notes | LoRA-SAM / SegFormer implementation (listed as later candidates only) |
| Unit tests for score-map assembly (mocked detections) | Guaranteeing SOTA numbers on first run |

## Model choice

- **Default:** `RFDETRSegMedium` (instance segmentation).
- Config key `variant` may select Nano / Small / Large later without a new trainer.
- License: Apache-2.0 (RF-DETR-Seg family). Prefer pinning an exact `rfdetr` version in `requirements.md`.

## Data

- Train/eval on the same glass core as E1: Track A **merged** YOLO-seg layout (`processed/glass_merged_seg` via `dataset: merged` + `weighting: balanced`).
- RF-DETR auto-detects YOLO-seg (`data.yaml` + image/label dirs). First smoke must prove our tree is accepted; if not, add a thin COCO export helper (fallback only — not the primary path).
- Class story matches E1: train **4-class** (`bubble`, `crack`, `inclusion`, `scratch`); at eval, **`linear_defect` = crack OR scratch** (same as YOLO).

## Training (`glass_crack_models/train/train_rfdetr.py`)

- CLI mirror of other trainers: `--config`, `--seed`, `--device`, `--smoke`, `--epochs`, `--fresh`, `--no-eval`, `--workers`, `--compute-units`.
- Resolve dataset dir from config (`dataset` / optional `data` yaml parent).
- Call `RFDETRSegMedium().train(dataset_dir=..., output_dir=run_dir, epochs=..., ...)` with batch settings suitable for TIDE L40 (document defaults in config; allow override).
- Run dir: `OUT_ROOT/runs/E7_s{seed}/` (or `{exp_id}_s{seed}`).
- Preferred eval checkpoint: `checkpoint_best_total.pth` (fallback: `checkpoint_best_ema.pth` / `checkpoint_best_regular.pth`).
- Resume: use package support if present; otherwise document `--fresh` for v1.
- After train (unless `--no-eval`): call `run_eval.main` on the preferred checkpoint.

### Config `e7_rfdetr_seg.yaml`

```yaml
exp_id: E7
track: C
variant: medium          # maps to RFDETRSegMedium
dataset: merged
weighting: balanced
epochs: 50               # DETR-style; shorter than YOLO 150 — tune after smoke
batch_size: 4
grad_accum_steps: 4
lr: 1.0e-4
seed: 0
notes: "Track C RF-DETR-Seg vs E1/E3"
```

Smoke: 1 epoch, tiny subset or RF-DETR’s minimal path, `imgsz`/resolution left to package defaults unless we must set `resolution=`.

## Inference (`RfDetrPredictor`)

- `track = "C"`.
- Load via `RFDETR.from_checkpoint(path, trust_checkpoint=True)` for our trusted local weights (or variant ctor + `pretrain_weights=` if `from_checkpoint` lacks metadata).
- `predict_batch(tiles)`:
  - Run RF-DETR predict at a low threshold (e.g. 0.01) so `run_eval` can sweep thresholds.
  - For each tile: score map = max over instances of `(confidence × mask)` for classes in the eval source set (crack|scratch → `linear_defect`; optional `crack` map).
  - Resize masks to tile H×W if needed.
- Register in `load_predictor`: detect Track C checkpoints (suffix/layout/`checkpoint_best_*.pth` or a small sidecar `experiment.json` with `track: C`). **Do not** steal Track B `.pth` smp checkpoints — use explicit markers (e.g. filename pattern + `experiment.json`, or only route when `RFDETR.from_checkpoint` succeeds and smp load is not intended).

**Routing rule (concrete):**

1. If sibling `experiment.json` has `"track": "C"` → RF-DETR.
2. Else if path name contains `checkpoint_best_` / `rfdetr` → try RF-DETR.
3. Else if suffix `.pth` → Semseg (B).
4. Else → YOLO (A).

## Evaluation / compare

- Same `run_eval` caches and `results.csv` columns.
- Notebook / `NEXT.md`: compare  
  `E1_s0 E1_s1 E1_s2 E3_s0 … E7_s0 …`  
  via `compare_runs --out …/yolo_unet_rfdetr`.
- Gate for *stacking YOLO recipes* remains E1; E7 is an **architecture compare**, not a prerequisite for E4/E4b.

## TIDE / laptop

- Install: `uv pip install rfdetr==<pin>` (and CUDA torch already present on TIDE).
- `tide_v4.ipynb` job cell: `MODULE=glass_crack_models.train.train_rfdetr`, `CFG=e7_rfdetr_seg.yaml`, `NAME=E7`, `DEV=cuda` or `0` as required by the wrapper.
- Laptop: CPU smoke only; full train on TIDE L40.

## Later model candidates (not in this scaffold)

| Candidate | Why | When |
|---|---|---|
| SegFormer (`mit_b2`) via Track B | Already supported by `train_semseg` arch/encoder | After E3 3-seed table |
| YOLO size sweep (`n`/`m` vs `s`) | Cheap Track A sensitivity | Optional ablation |
| LoRA-SAM / CrackSAM-style | Domain shift / thin cracks | Only if proxies + RF-DETR still miss curtain-wall |
| RF-DETR-Seg Nano/Large | Accuracy–latency knobs | After Medium baseline |

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `rfdetr` API / checkpoint layout drift | Pin version; smoke test; prefer `from_checkpoint` |
| YOLO-seg folder not auto-detected | Smoke on real `glass_merged_seg`; COCO export fallback only if needed |
| VRAM on L40 | Default batch 4 × accum 4; document OOM knobs |
| Checkpoint routing vs Track B `.pth` | `experiment.json` track marker + filename heuristics |
| Early RF-DETR results look weak | Report honestly under same protocol; don’t change matching |

## Success criteria

1. `python -m glass_crack_models.train.train_rfdetr --config e7_rfdetr_seg.yaml --smoke` completes without protocol changes.
2. A finished E7 checkpoint scores through `run_eval` and appears in `compare_runs` beside E1/E3.
3. Docs state clearly how to launch Track C on TIDE and what is intentionally deferred.

## Implementation order (for the plan)

1. Pin `rfdetr`; add E7 config.  
2. `RfDetrPredictor` + `load_predictor` routing + unit test (mock masks).  
3. `train_rfdetr.py` + smoke.  
4. Wire `NEXT.md` / `tide_v4` / `requirements.md`.  
5. Optional: one real TIDE E7_s0 after smoke green.
