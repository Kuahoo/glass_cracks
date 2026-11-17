#!/usr/bin/env bash
# One-time (and safe to re-run) setup on the SDSU TIDE JupyterHub, PyTorch Notebook image.
#   bash scripts/tide/setup.sh
# 1. Python venv in $HOME that reuses the image's CUDA torch, plus the pinned packages.
# 2. Unzip $GC_ROOT/bundle_colab.zip, then the optional extras.zip, into $DATA_ROOT (first run only).
# 3. Rebuild the machine-specific dataset files (relabel_classes) and print a session report.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$GC_ROOT" "$OUT_ROOT" "$TORCH_HOME" "$HF_HOME" "$YOLO_CONFIG_DIR"

PINS=(ultralytics==8.4.177 segmentation-models-pytorch==0.5.0 timm==1.0.30 transformers==5.19.0
      albumentations==2.0.8 pycocotools==2.0.11 scikit-image==0.26.0 shapely==2.2.0
      'rfdetr[train]==1.12.1')
STAMP="$VENV/.pins"

if [ ! -x "$PY" ]; then
    echo "== creating venv $VENV (inherits the image's torch)"
    python3 -m venv --system-site-packages "$VENV"
fi
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "${PINS[*]}" ]; then
    echo "== installing pinned packages"
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q "${PINS[@]}"
    echo "${PINS[*]}" > "$STAMP"
fi
"$PY" - <<'EOF'
import torch
if not torch.cuda.is_available():
    print("WARNING: torch sees no GPU. Request a GPU on the spawn page (PyTorch Notebook image).")
else:
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  gpu {torch.cuda.get_device_name(0)}")
EOF

# Validate a zip (an upload cut short or a saved Drive page is not one), then extract it into $DATA_ROOT.
unzip_into_data() {
    if ! "$PY" -c "import sys, zipfile; sys.exit(not zipfile.is_zipfile(sys.argv[1]))" "$1"; then
        echo "ERROR: $1 is not a valid zip ($(stat -c %s "$1") bytes)." >&2
        head -c 200 "$1" | grep -qi "<html\|<!doctype" && echo "  It is an HTML page: a Drive link saved the download page, not the file." >&2
        echo "  Delete it and upload again (wait for the upload bar to finish), or fetch it with gdown." >&2
        exit 1
    fi
    echo "== unzipping $1 -> $DATA_ROOT"
    mkdir -p "$DATA_ROOT"
    "$PY" -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$1" "$DATA_ROOT"
}

if [ ! -f "$DATA_ROOT/processed/glass_merged_seg/manifest.csv" ]; then
    ZIP="$GC_ROOT/bundle_colab.zip"
    if [ ! -f "$ZIP" ]; then
        echo "ERROR: upload bundle_colab.zip to $ZIP first (JupyterLab upload button), then re-run." >&2
        exit 1
    fi
    unzip_into_data "$ZIP"
fi

# Plan v4 extras (scripts/build_extras.sh on the laptop): CUBIT tiles, CarDD subset, hard negatives.
# Needed only by E4b, E5 and E6. Upload extras.zip to $GC_ROOT, or set EXTRAS_URL to a Google Drive link.
if [ ! -f "$DATA_ROOT/extras/cubit_tiles/data.yaml" ]; then
    EXTRAS="$GC_ROOT/extras.zip"
    if [ ! -f "$EXTRAS" ] && [ -n "${EXTRAS_URL:-}" ]; then
        "$PY" -m pip install -q gdown && "$PY" -m gdown --fuzzy "$EXTRAS_URL" -O "$EXTRAS"
    fi
    if [ -f "$EXTRAS" ]; then
        unzip_into_data "$EXTRAS" && rm "$EXTRAS"
    else
        echo "NOTE: no $EXTRAS yet; E1-E4 and the R0-R2 ablation run without it, E4b/E5/E6 need it."
    fi
fi

cd "$CODE_DIR"
echo "== rebuilding linear_defect set and weighting lists"
"$PY" -m glass_crack_models.data.relabel_classes
"$PY" -c "from ultralytics.data.utils import check_det_dataset as c; d = c('$DATA_ROOT/processed/glass_merged_linear/data_balanced.yaml'); print('dataset ok:', d['names'])"

echo
echo "== session: GPU ${GPU_NAME:-none} | CPUs $GC_CPUS | /dev/shm ${GC_SHM_MB:-?} MB -> WORKERS=$WORKERS"
[ "$WORKERS" = 0 ] && echo "   /dev/shm is small, so data loading runs in the main process (slow). Ask TIDE support for a larger /dev/shm."
df -h "$HOME" | tail -1 | awk '{print "== home storage: " $3 " used of " $2}'
