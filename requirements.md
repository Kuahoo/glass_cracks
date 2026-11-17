# Python requirements

Environment: `uv`-managed virtualenv at `.venv/` (Python 3.13.7). Versions below are what is installed and tested as of 2026-10-13.

## Laptop (CPU only)

The laptop has no CUDA GPU (integrated Radeon 680M), so PyTorch comes from the CPU wheel index. Use it for code, unit tests, data prep (CUBIT tiling) and small `predict.py` runs.

```sh
uv venv --python 3.13 .venv            # only if .venv does not exist yet
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.14.1 torchvision==0.29.1
uv pip install --python .venv/bin/python \
  ultralytics==8.4.177 \
  segmentation-models-pytorch==0.5.0 timm==1.0.30 \
  transformers==5.14.1 \
  albumentations==2.0.8 \
  opencv-python==5.0.0.93 \
  pycocotools==2.0.11 scikit-image==0.26.0 shapely==2.2.0 scipy==1.18.1 \
  numpy==2.5.4 pandas==3.0.6 pillow==12.3.0 pyyaml==6.0.3 \
  roboflow==1.7.1 gdown==6.4.2
```

## Google Colab (GPU)

Colab already ships a CUDA build of `torch`/`torchvision`; do not reinstall them. Record the preinstalled versions in each run's log instead. Prefer `transformers==5.14.1` unless Colab torch is ≥2.6 (then 5.19 is fine).

```sh
pip install -q \
  ultralytics==8.4.177 \
  segmentation-models-pytorch==0.5.0 timm==1.0.30 \
  transformers==5.14.1 \
  albumentations==2.0.8 \
  pycocotools==2.0.11 scikit-image==0.26.0 shapely==2.2.0 \
  "rfdetr[train]==1.12.1"
```

## SDSU TIDE JupyterHub (GPU)

Use the **PyTorch Notebook** image (ships `torch 2.4.0+cu121`). `scripts/tide/setup.sh` creates `~/glass_crack/venv` with `--system-site-packages`, then installs into the venv:

- `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, `torchaudio==2.5.1+cu121`
- `transformers==5.14.1`, `rfdetr[train]==1.12.1`

**Why not transformers 5.19?** It needs `torch.accelerator` → **torch ≥2.6** (cu124+ wheels). On cu121 we stay at 2.5.1 + 5.14.1 (verified: `AutoBackbone` + `import rfdetr` OK). Needs ~2 GB free under `$HOME`.

```sh
bash ~/glass_crack/cracked/scripts/tide/setup.sh
```

## What each package is for

| Package | Used for |
|---|---|
| `torch`, `torchvision`, `torchaudio` | Training and inference (TIDE venv: **2.5.1+cu121**) |
| `ultralytics` | Track A: YOLO-seg training, validation, `data.yaml` loading |
| `segmentation-models-pytorch`, `timm` | Track B: U-Net (ResNet34 / EfficientNet encoders) and SegFormer (`mit_b2`) |
| `transformers` | SegFormer + Track C DINOv2. Pin **`5.14.1`** with torch 2.5; use 5.19 only if torch≥2.6 |
| `rfdetr[train]` | Track C: RF-DETR-Seg training + inference (`transformers>=5.1,<6`) |
| `albumentations` | Track B augmentation; Phase 4 scale/blur/compression augmentation |
| `opencv-python` | Polygon rasterization, connected components, overlays, tiling |
| `pycocotools` | Mask RLE encoding and COCO-style mask IoU in the evaluator |
| `scikit-image` | Skeletonization for clDice and crack length |
| `shapely` | Polygon clipping and simplification when tiling CUBIT |
| `scipy` | Bootstrap statistics |
| `numpy`, `pandas`, `pillow`, `pyyaml` | Arrays, `results/results.csv`, image I/O, configs |
| `roboflow`, `gdown` | Dataset download (`temp/download_datasets.py`) |

Licenses are deferred (see `model_implementation_plan.md`, Section 9); note that `ultralytics` is AGPL-3.0.
