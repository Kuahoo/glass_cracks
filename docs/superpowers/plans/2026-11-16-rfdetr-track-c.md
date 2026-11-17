# Track C RF-DETR-Seg Large Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold Track C so RF-DETR-Seg Large trains on the glass YOLO-seg set and scores through the same `run_eval` / `compare_runs` path as YOLO (E1) and U-Net (E3).

**Architecture:** Thin adapter around the official `rfdetr` package. `train_rfdetr.py` wraps `RFDETRSegLarge.train()`; `RfDetrPredictor` turns instance masks×scores into per-class score maps; `load_predictor` routes Track C via `experiment.json` / checkpoint name heuristics without breaking Track B `.pth` loads.

**Tech Stack:** Python 3.11–3.13, PyTorch (TIDE image CUDA), `rfdetr[train]`, existing `glass_crack_models` eval protocol, Ultralytics-format glass merged dataset.

**Spec:** `docs/superpowers/specs/2026-11-16-rfdetr-track-c-design.md`

## Global Constraints

- Default variant is **Large** (`RFDETRSegLarge`) for TIDE L40 / 8 CPU / 32 GB RAM.
- Default train knobs: `batch_size: 2`, `grad_accum_steps: 8`, `epochs: 50`, `lr: 1.0e-4`, `workers: 8`.
- Dataset: Track A **merged** 4-class glass set; eval `linear_defect` = crack OR scratch (same as YOLO).
- Do not change matching / threshold-free protocol.
- Pin `rfdetr[train]` to an exact version discovered at implement time (`pip index versions rfdetr` or PyPI); record it in `requirements.md`.
- Checkpoint for eval: prefer `checkpoint_best_total.pth`, then `checkpoint_best_ema.pth`, then `checkpoint_best_regular.pth`.
- Commits on branch `models`; do not push unless asked.

## File map

| File | Role |
|---|---|
| `glass_crack_models/configs/e7_rfdetr_seg.yaml` | E7 Large config |
| `glass_crack_models/infer/predictors.py` | Add `RfDetrPredictor`, extend `load_predictor` |
| `glass_crack_models/train/train_rfdetr.py` | Train + optional `run_eval` hook |
| `glass_crack_models/tests/test_rfdetr_predictor.py` | Score-map assembly + routing tests (mock rfdetr) |
| `requirements.md` | Pin + package table row |
| `NEXT.md`, `notebooks/tide_v4.ipynb`, `README.md` | Launch / compare docs |

---

### Task 1: E7 config + dependency pin

**Files:**
- Create: `glass_crack_models/configs/e7_rfdetr_seg.yaml`
- Modify: `requirements.md`

**Interfaces:**
- Produces: config keys `exp_id`, `track`, `variant`, `dataset`, `weighting`, `epochs`, `batch_size`, `grad_accum_steps`, `lr`, `seed`, `workers`, `notes` for Task 3

- [ ] **Step 1: Resolve an exact `rfdetr` version**

```bash
pip index versions rfdetr 2>/dev/null | head -5
# or: uv pip install "rfdetr[train]" --dry-run 2>&1 | head
```

Record the chosen version as `RFDETR_PIN` (example shape `1.8.2` — use whatever is current stable ≥1.8.2).

- [ ] **Step 2: Write `e7_rfdetr_seg.yaml`**

```yaml
# E7 (Track C): RF-DETR-Seg Large on the 4-class glass core. Compare with E1 / E3 under run_eval.
exp_id: E7
track: C
variant: large
dataset: merged
weighting: balanced
epochs: 50
batch_size: 2
grad_accum_steps: 8
lr: 1.0e-4
seed: 0
workers: 8
notes: "Track C RF-DETR-Seg Large vs E1/E3"
```

- [ ] **Step 3: Update `requirements.md`**

Add install line under TIDE/Colab sections:

```sh
uv pip install --python .venv/bin/python "rfdetr[train]==<RFDETR_PIN>"
```

Add table row: `| \`rfdetr[train]\` | Track C: RF-DETR-Seg training + inference |`.

- [ ] **Step 4: Commit**

```bash
git add glass_crack_models/configs/e7_rfdetr_seg.yaml requirements.md
git commit -m "Add E7 RF-DETR-Seg Large config and dependency pin."
```

---

### Task 2: `RfDetrPredictor` + `load_predictor` routing (TDD)

**Files:**
- Modify: `glass_crack_models/infer/predictors.py`
- Create: `glass_crack_models/tests/test_rfdetr_predictor.py`

**Interfaces:**
- Consumes: `eval_class_sources(class_names)` from `predictors.py`
- Produces:
  - `class RfDetrPredictor` with `track = "C"`, `predict_batch(tiles: list[np.ndarray]) -> list[dict[str, np.ndarray]]`
  - `def prefer_rfdetr_checkpoint(run_dir: Path) -> Path`
  - `load_predictor(checkpoint, device=None)` routes Track C before generic `.pth` → Semseg

- [ ] **Step 1: Write failing tests**

Create `glass_crack_models/tests/test_rfdetr_predictor.py`:

```python
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from glass_crack_models.infer.predictors import (
    RfDetrPredictor,
    load_predictor,
    prefer_rfdetr_checkpoint,
    score_maps_from_instances,
)


class ScoreMapTests(unittest.TestCase):
    def test_max_conf_times_mask_for_linear_defect(self):
        h, w = 8, 8
        m0 = np.zeros((h, w), bool); m0[2:5, 2:5] = True
        m1 = np.zeros((h, w), bool); m1[4:7, 4:7] = True
        maps = score_maps_from_instances(
            height=h, width=w,
            masks=np.stack([m0, m1]),
            confs=np.array([0.9, 0.4], np.float32),
            class_ids=np.array([1, 3], np.int64),  # crack, scratch
            class_names=["bubble", "crack", "inclusion", "scratch"],
        )
        self.assertIn("linear_defect", maps)
        self.assertIn("crack", maps)
        self.assertAlmostEqual(float(maps["linear_defect"][3, 3]), 0.9, places=5)
        self.assertAlmostEqual(float(maps["linear_defect"][5, 5]), 0.4, places=5)
        self.assertAlmostEqual(float(maps["crack"][3, 3]), 0.9, places=5)
        self.assertEqual(float(maps["crack"][5, 5]), 0.0)


class RoutingTests(unittest.TestCase):
    def test_prefer_checkpoint_order(self, tmp_path=None):
        root = Path(self.id())  # unused; use tempfile below
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "checkpoint_best_regular.pth").write_bytes(b"a")
            (d / "checkpoint_best_ema.pth").write_bytes(b"b")
            (d / "checkpoint_best_total.pth").write_bytes(b"c")
            self.assertEqual(prefer_rfdetr_checkpoint(d).name, "checkpoint_best_total.pth")

    def test_load_predictor_uses_track_c_experiment_json(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            ck = d / "checkpoint_best_total.pth"
            ck.write_bytes(b"x")
            (d / "experiment.json").write_text(json.dumps({"track": "C", "variant": "large"}))
            with patch("glass_crack_models.infer.predictors.RfDetrPredictor") as cls:
                cls.return_value = MagicMock(track="C")
                pred = load_predictor(ck, device="cpu")
                cls.assert_called_once()
                self.assertEqual(pred.track, "C")


if __name__ == "__main__":
    unittest.main()
```

Fix the tempfile usage in `test_prefer_checkpoint_order` to only use `TemporaryDirectory` (remove the bogus `Path(self.id())` line when implementing).

- [ ] **Step 2: Run tests — expect FAIL**

```bash
# from repo root, with a working venv that has numpy
python -m unittest glass_crack_models.tests.test_rfdetr_predictor -v
```

Expected: `ImportError` or `AttributeError` for missing `score_maps_from_instances` / `RfDetrPredictor`.

- [ ] **Step 3: Implement helpers + predictor in `predictors.py`**

Add (near other predictors):

```python
def score_maps_from_instances(
    height: int,
    width: int,
    masks: np.ndarray,
    confs: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str],
) -> dict[str, np.ndarray]:
    sources = eval_class_sources(class_names)
    maps = {c: np.zeros((height, width), np.float32) for c in sources}
    if masks.size == 0:
        return maps
    if masks.shape[-2:] != (height, width):
        masks = np.stack([cv2.resize(m.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) > 0.5
                          for m in masks])
    for name, ids in sources.items():
        sel = np.isin(class_ids, ids)
        if sel.any():
            maps[name] = np.max(masks[sel].astype(np.float32) * confs[sel, None, None], axis=0).astype(np.float32)
    return maps


def prefer_rfdetr_checkpoint(run_dir: Path) -> Path:
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no RF-DETR best checkpoint under {run_dir}")


def _experiment_track(checkpoint: Path) -> str | None:
    for side in (checkpoint.parent / "experiment.json", checkpoint.with_name("experiment.json")):
        if side.exists():
            try:
                return json.loads(side.read_text()).get("track")
            except Exception:
                return None
    return None


class RfDetrPredictor:
    track = "C"

    def __init__(self, checkpoint: str | Path, device: str | None = None, conf: float = 0.01):
        from rfdetr import RFDETR
        self.checkpoint = str(checkpoint)
        self.device = torch_device(device) if device and device[0].isdigit() else (device or default_device())
        self.conf = conf
        # trust_checkpoint: local training artifacts we produced
        self.model = RFDETR.from_checkpoint(self.checkpoint, trust_checkpoint=True)
        names = getattr(self.model, "class_names", None) or getattr(self.model, "names", None)
        if names is None:
            raise RuntimeError(f"RF-DETR checkpoint missing class names: {self.checkpoint}")
        if isinstance(names, dict):
            self.class_names = [names[i] for i in sorted(names)]
        else:
            self.class_names = list(names)
        self.sources = eval_class_sources(self.class_names)
        self.eval_classes = list(self.sources)

    def predict_batch(self, tiles: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        out = []
        for tile in tiles:
            h, w = tile.shape[:2]
            det = self.model.predict(tile, threshold=self.conf)
            masks = getattr(det, "mask", None)
            if masks is None or len(masks) == 0:
                out.append({c: np.zeros((h, w), np.float32) for c in self.eval_classes})
                continue
            confs = np.asarray(det.confidence, dtype=np.float32)
            class_ids = np.asarray(det.class_id, dtype=np.int64)
            out.append(score_maps_from_instances(h, w, np.asarray(masks), confs, class_ids, self.class_names))
        return out
```

Update `load_predictor`:

```python
def load_predictor(checkpoint: str | Path, device: str | None = None):
    import json
    p = Path(checkpoint)
    track = _experiment_track(p)
    name = p.name.lower()
    if track == "C" or name.startswith("checkpoint_best_") or "rfdetr" in name:
        return RfDetrPredictor(p, device)
    if p.suffix == ".pth":
        return SemsegPredictor(p, device)
    return YoloPredictor(p, device)
```

Add `import json` at top of `predictors.py` if missing.

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m unittest glass_crack_models.tests.test_rfdetr_predictor -v
```

Expected: all OK (routing test mocks `RfDetrPredictor`; score-map test needs no `rfdetr` install).

- [ ] **Step 5: Commit**

```bash
git add glass_crack_models/infer/predictors.py glass_crack_models/tests/test_rfdetr_predictor.py
git commit -m "Add RF-DETR predictor and Track C checkpoint routing."
```

---

### Task 3: `train_rfdetr.py` + smoke path

**Files:**
- Create: `glass_crack_models/train/train_rfdetr.py`
- Test: manual smoke command (and optional unittest that `variant_class("large")` maps correctly without GPU)

**Interfaces:**
- Consumes: `e7_rfdetr_seg.yaml`, `paths.merged_dir`, `prefer_rfdetr_checkpoint`, `run_eval.main`
- Produces: `OUT_ROOT/runs/<exp_id>_s<seed>/` with `experiment.json` containing `"track": "C"` and RF-DETR checkpoints

- [ ] **Step 1: Write variant map helper test (failing)**

Add to `test_rfdetr_predictor.py` or new `test_train_rfdetr.py`:

```python
def test_variant_class_large():
    from glass_crack_models.train.train_rfdetr import variant_class
    self.assertEqual(variant_class("large").__name__, "RFDETRSegLarge")
```

(Skip entire test module if `import rfdetr` fails with a clear skip reason — score-map tests must still run.)

- [ ] **Step 2: Implement `train_rfdetr.py`**

```python
"""Track C: RF-DETR-Seg training (E7) with run_eval hook.

    python -m glass_crack_models.train.train_rfdetr --config e7_rfdetr_seg.yaml [--seed 0]
    python -m glass_crack_models.train.train_rfdetr --config e7_rfdetr_seg.yaml --smoke
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from glass_crack_models import paths
from glass_crack_models.config import load_config
from glass_crack_models.infer.predictors import prefer_rfdetr_checkpoint

DATASETS = {"linear": paths.linear_dir, "merged": paths.merged_dir}

VARIANTS = {
    "nano": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
}


def variant_class(name: str):
    import rfdetr
    key = name.lower().strip()
    if key not in VARIANTS:
        raise KeyError(f"unknown RF-DETR variant {name!r}; choose from {sorted(VARIANTS)}")
    return getattr(rfdetr, VARIANTS[key])


def dataset_dir(cfg: dict) -> Path:
    if "data" in cfg:
        y = paths.resolve(cfg["data"])
        return y.parent if y.suffix in {".yaml", ".yml"} else y
    ds = DATASETS[cfg["dataset"]]()
    if not (ds / "data_all.yaml").exists() and not (ds / "data_balanced.yaml").exists():
        raise FileNotFoundError(f"{ds} missing data_*.yaml — run relabel_classes / ensure glass merged set")
    return ds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--compute-units", default="")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = int(cfg.get("seed", 0))
    name = f"{cfg['exp_id']}_s{seed}" + ("_smoke" if args.smoke else "")
    run_dir = paths.out_root() / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment.json").write_text(json.dumps({**cfg, "track": "C"}, indent=1, default=str))

    epochs = 1 if args.smoke else int(cfg.get("epochs", 50))
    if args.epochs is not None:
        epochs = args.epochs
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 8))
    batch = int(cfg.get("batch_size", 2))
    accum = int(cfg.get("grad_accum_steps", 8))
    if args.smoke:
        batch, accum, workers = 1, 1, 0

    ds = dataset_dir(cfg)
    cls = variant_class(cfg.get("variant", "large"))
    model = cls()
    # device: leave package default unless we need to force CPU for smoke
    train_kw = dict(
        dataset_dir=str(ds),
        output_dir=str(run_dir),
        epochs=epochs,
        batch_size=batch,
        grad_accum_steps=accum,
        lr=float(cfg.get("lr", 1e-4)),
        num_workers=workers,
    )
    if args.fresh:
        # package-specific: if resume exists, caller should delete run_dir weights; document in NEXT.md
        pass
    model.train(**train_kw)

    if args.no_eval:
        return
    ck = prefer_rfdetr_checkpoint(run_dir)
    from glass_crack_models.eval import run_eval
    ev = ["--checkpoint", str(ck), "--exp-id", cfg["exp_id"], "--seed", str(seed),
          "--task", f"rfdetr-seg-{cfg.get('variant', 'large')}",
          "--pretrain", "coco", "--weighting", cfg.get("weighting", ""),
          "--compute-units", args.compute_units, "--notes", cfg.get("notes", "")]
    if args.device:
        ev += ["--device", args.device]
    if args.smoke:
        ev += ["--limit", "12", "--bootstrap", "200", "--out", str(paths.out_root() / "evals" / name)]
    run_eval.main(ev)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
```

Adjust `model.train(...)` kwargs to match the pinned `rfdetr` version’s signature if names differ (`num_workers` vs `workers`). Check with:

```bash
python -c "import inspect, rfdetr; from rfdetr import RFDETRSegLarge; print(inspect.signature(RFDETRSegLarge.train))"
```

- [ ] **Step 3: Install pin in the active env and run smoke (GPU preferred; CPU only if smoke supports integer batch)**

```bash
uv pip install --python .venv/bin/python "rfdetr[train]==<RFDETR_PIN>"
# DATA_ROOT must point at the glass bundle that contains processed/glass_merged_seg
python -m glass_crack_models.train.train_rfdetr --config e7_rfdetr_seg.yaml --smoke --no-eval
```

Expected: training starts, writes under `outputs/runs/E7_s0_smoke/`, exits 0. If YOLO-seg layout is rejected, stop and add a COCO export helper (spec fallback) before continuing — do not silently train on the wrong path.

- [ ] **Step 4: Commit**

```bash
git add glass_crack_models/train/train_rfdetr.py glass_crack_models/tests/test_rfdetr_predictor.py
git commit -m "Add train_rfdetr Track C trainer with smoke path."
```

---

### Task 4: Docs + TIDE compare wiring

**Files:**
- Modify: `NEXT.md`
- Modify: `notebooks/tide_v4.ipynb`
- Modify: `README.md` (Track C bullet under §1 / configs table)

**Interfaces:**
- Consumes: E7 launch command from Task 3
- Produces: operator instructions for YOLO vs U-Net vs RF-DETR compare

- [ ] **Step 1: Update `NEXT.md`**

Add section after U-Net:

```markdown
### Track C — RF-DETR-Seg Large (architecture compare)
Requires `rfdetr[train]` in the TIDE venv. Launch with `train_rfdetr` (not `train_yolo`):

CFG=e7_rfdetr_seg.yaml; NAME=E7; SEED=0
MODULE=glass_crack_models.train.train_rfdetr
bash "$CODE_DIR/scripts/tide/job.sh" start ${NAME}_s$SEED $MODULE \
  --config $CFG --seed $SEED --device cuda --workers "$WORKERS" \
  --compute-units "TIDE $GPU_NAME"

Compare when seeds exist:
"$PY" -m glass_crack_models.eval.compare_runs \
  --runs E1_s0 E1_s1 E1_s2 E3_s0 E3_s1 E3_s2 E7_s0 \
  --out "$OUT_ROOT/results/yolo_unet_rfdetr"
```

- [ ] **Step 2: Update `tide_v4.ipynb` training table + job cell comment**

Add row: `| Track C | e7_rfdetr_seg.yaml | E7 | train_rfdetr | 0 (then 1–2) |`  
Uncomment example:

```bash
# CFG=e7_rfdetr_seg.yaml; NAME=E7; SEED=0; MODULE=glass_crack_models.train.train_rfdetr; DEV=cuda
```

Extend compare cell to include `E7_s0` when present.

- [ ] **Step 3: README Track C bullet**

Under §1 tracks:

```markdown
- **Track C:** RF-DETR-Seg (`e7_rfdetr_seg.yaml`, default Large). Trained via `train_rfdetr.py`; scored with the same `run_eval` protocol for architecture compare vs E1/E3.
```

- [ ] **Step 4: Commit**

```bash
git add NEXT.md notebooks/tide_v4.ipynb README.md
git commit -m "Document Track C RF-DETR-Seg Large launch and compare."
```

---

## Spec coverage check

| Spec item | Task |
|---|---|
| E7 Large config + batch 2×accum 8 | Task 1 |
| `rfdetr` pin in requirements | Task 1 |
| `RfDetrPredictor` + score maps | Task 2 |
| `load_predictor` Track C routing | Task 2 |
| `prefer_rfdetr_checkpoint` | Task 2 |
| `train_rfdetr.py` + smoke | Task 3 |
| `experiment.json` track C | Task 3 |
| `run_eval` hook | Task 3 |
| NEXT / tide_v4 / README | Task 4 |
| SegFormer/SAM later only | Documented in spec; not implemented |
| COCO fallback | Only if smoke proves YOLO layout fails (explicit stop in Task 3) |

## Placeholder scan

No TBD/TODO steps. Exact paths, commands, and code included. Adjust only `RFDETR_PIN` and `train()` kwarg names after inspecting the pinned package.
