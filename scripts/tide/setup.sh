#!/usr/bin/env bash
# One-time (and safe to re-run) setup on the SDSU TIDE JupyterHub, PyTorch Notebook image.
#   bash scripts/tide/setup.sh
# 1. Python venv in $HOME. Image ships torch 2.4; we install torch 2.5.1+cu121 into the venv
#    (overrides system-site-packages) so transformers/rfdetr work.
# 2. Unzip $GC_ROOT/bundle_colab.zip, then the optional extras.zip, into $DATA_ROOT (first run only).
# 3. Rebuild the machine-specific dataset files (relabel_classes) and print a session report.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$GC_ROOT" "$OUT_ROOT" "$TORCH_HOME" "$HF_HOME" "$YOLO_CONFIG_DIR"

# Image default is torch 2.4.0+cu121. Track C needs:
#   - torch>=2.5 for DTensor (used by transformers>=5.14)
#   - transformers>=5.1 for BackboneConfigMixin / AutoBackbone (rfdetr)
# transformers 5.19 needs torch.accelerator → torch>=2.6 (cu124+). Stay on cu121 torch 2.5.1
# + transformers 5.14.1 (verified: AutoBackbone + rfdetr import OK).
TORCH_INDEX=https://download.pytorch.org/whl/cu121
TORCH_PINS=(torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121)
TRANSFORMERS_PIN=transformers==5.14.1
PINS=(ultralytics==8.4.177 segmentation-models-pytorch==0.5.0 timm==1.0.30 "$TRANSFORMERS_PIN"
      albumentations==2.0.8 pycocotools==2.0.11 scikit-image==0.26.0 shapely==2.2.0
      'rfdetr[train]==1.12.1')
STAMP="$VENV/.pins"
STAMP_BODY="${TORCH_PINS[*]} | ${PINS[*]}"

if [ ! -x "$PY" ]; then
    echo "== creating venv $VENV (system-site-packages; venv torch will override image 2.4)"
    python3 -m venv --system-site-packages "$VENV"
fi
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$STAMP_BODY" ]; then
    echo "== installing ${TORCH_PINS[*]} from $TORCH_INDEX (~2 GB; needs free home space)"
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q --index-url "$TORCH_INDEX" "${TORCH_PINS[@]}"
    echo "== installing pinned packages ($TRANSFORMERS_PIN + rfdetr)"
    "$PY" -m pip install -q "${PINS[@]}"
    # Keep CUDA torch if a PyPI dep tried to replace it; keep transformers pin if rfdetr upgraded it.
    if ! "$PY" -c "import torch; v=torch.__version__; raise SystemExit(0 if v.startswith('2.5.1') and 'cu121' in v else 1)"; then
        echo "== re-applying ${TORCH_PINS[*]}"
        "$PY" -m pip install -q --index-url "$TORCH_INDEX" "${TORCH_PINS[@]}"
    fi
    "$PY" -m pip install -q --force-reinstall --no-deps "$TRANSFORMERS_PIN"
    echo "$STAMP_BODY" > "$STAMP"
fi

# Fail loudly if torch/transformers/rfdetr are still incompatible.
"$PY" - <<'EOF'
import importlib.metadata as md
import torch
from packaging.version import Version

def ver(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "missing"

def causes(err):
    lines = [f"{type(err).__name__}: {err}"]
    c = err.__cause__ or getattr(err, "__context__", None)
    n = 0
    while c is not None and n < 6:
        lines.append(f"  caused by {type(c).__name__}: {c}")
        c = c.__cause__ or getattr(c, "__context__", None)
        n += 1
    return "\n".join(lines)

tv, tr = torch.__version__, ver("transformers")
print(f"torch {tv}  cuda {getattr(torch.version, 'cuda', None)}  "
      f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print(f"torch file: {torch.__file__}")
print(f"transformers {tr}  rfdetr {ver('rfdetr')}  torchaudio {ver('torchaudio')}")
if not torch.cuda.is_available():
    print("WARNING: torch sees no GPU. Request a GPU on the spawn page (PyTorch Notebook image).")
if "/opt/conda/" in (torch.__file__ or "") or Version(tv.split("+")[0]) < Version("2.5"):
    raise SystemExit(
        f"ERROR: need venv torch 2.5.1+cu121; got {tv} from {torch.__file__}.\n"
        f"  Fix: rm -f ~/glass_crack/venv/.pins && bash ~/glass_crack/cracked/scripts/tide/setup.sh"
    )
if not tr.startswith("5.14."):
    raise SystemExit(
        f"ERROR: need transformers 5.14.x on torch 2.5 (5.19 needs torch>=2.6 / torch.accelerator); got {tr}.\n"
        f"  Fix: pip install --force-reinstall --no-deps 'transformers==5.14.1' && rm -f ~/glass_crack/venv/.pins"
    )
try:
    from torch.distributed.tensor import DTensor  # noqa: F401
except ImportError as e:
    raise SystemExit(f"ERROR: torch {tv} missing DTensor:\n{causes(e)}") from e
try:
    from transformers.utils import is_torch_available
    from transformers import AutoBackbone, BackboneConfigMixin  # noqa: F401
    assert is_torch_available(), "transformers disabled PyTorch"
except Exception as e:
    raise SystemExit(f"ERROR: transformers import failed:\n{causes(e)}") from e
try:
    import rfdetr  # noqa: F401
except Exception as e:
    raise SystemExit(f"ERROR: import rfdetr failed:\n{causes(e)}") from e
print("Track C imports OK (torch 2.5.1, transformers 5.14, AutoBackbone, rfdetr)")
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
