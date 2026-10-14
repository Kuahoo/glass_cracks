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
  transformers==5.19.0 \
  albumentations==2.0.8 \
  opencv-python==5.0.0.93 \
  pycocotools==2.0.11 scikit-image==0.26.0 shapely==2.2.0 scipy==1.18.1 \
  numpy==2.5.4 pandas==3.0.6 pillow==12.3.0 pyyaml==6.0.3 \
  roboflow==1.7.1 gdown==6.4.2
```

## Google Colab (GPU)

Colab already ships a CUDA build of `torch`/`torchvision`; do not reinstall them. Record the preinstalled versions in each run's log instead.

```sh
pip install -q \
  ultralytics==8.4.177 \
  segmentation-models-pytorch==0.5.0 timm==1.0.30 \
  transformers==5.19.0 \
  albumentations==2.0.8 \
  pycocotools==2.0.11 scikit-image==0.26.0 shapely==2.2.0
```

## What each package is for

| Package | Used for |
|---|---|
| `torch`, `torchvision` | Training and inference |
| `ultralytics` | Track A: YOLO-seg training, validation, `data.yaml` loading |
| `segmentation-models-pytorch`, `timm` | Track B: U-Net (ResNet34 / EfficientNet encoders) and SegFormer (`mit_b2`) |
| `transformers` | Alternative SegFormer implementation and weights |
| `albumentations` | Track B augmentation; Phase 4 scale/blur/compression augmentation |
| `opencv-python` | Polygon rasterization, connected components, overlays, tiling |
| `pycocotools` | Mask RLE encoding and COCO-style mask IoU in the evaluator |
| `scikit-image` | Skeletonization for clDice and crack length |
| `shapely` | Polygon clipping and simplification when tiling CUBIT |
| `scipy` | Bootstrap statistics |
| `numpy`, `pandas`, `pillow`, `pyyaml` | Arrays, `results/results.csv`, image I/O, configs |
| `roboflow`, `gdown` | Dataset download (`temp/download_datasets.py`) |

Licenses are deferred (see `model_implementation_plan.md`, Section 9); note that `ultralytics` is AGPL-3.0.
