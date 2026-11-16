#!/usr/bin/env bash
# Laptop: build the plan-v4 extra datasets from the raw sources in data/ and pack data/extras.zip for TIDE.
#   bash scripts/build_extras.sh            # ~10 min with 8 workers; zip ~2.3 GB
# data/extras/
#   cubit_tiles/  CUBIT-InSeg CR1 crack tiles, 1024 px windows -> 640, class linear_defect  (E5 pretraining)
#   cardd/        CarDD `crack` + `glass shatter`, single class                             (E6 pretraining)
#   hardneg/      crack-free curtain-wall / CUBIT / facade tiles, half with synthetic glare (E4b)
# Upload extras.zip to ~/glass_crack/ on TIDE (or Drive + EXTRAS_URL) and re-run scripts/tide/setup.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-.venv/bin/python}"
[ -d data/curtain_wall/dataset ] || git clone --depth 1 \
    https://github.com/Brandon-Johns/glass-curtain-wall-installation-dataset.git data/curtain_wall
"$PY" -m glass_crack_models.data.tile_cubit --src data/cubit_inseg --out data/extras/cubit_tiles \
    --windows 1024 --jpeg-quality 85 --workers "${WORKERS:-8}"
"$PY" phase_0/prepare_dataset_bundle.py --data-root data --only cardd --out-dir data/extras/cardd
"$PY" phase_0/prepare_dataset_bundle.py --data-root data --only neg_cwall neg_cubit neg_fcd \
    --reflect-frac 0.5 --val-ratio 0 --out-dir data/extras/hardneg
(cd data && rm -f extras.zip && zip -qr -0 extras.zip extras)   # JPEGs: store, do not recompress
ls -l data/extras.zip
