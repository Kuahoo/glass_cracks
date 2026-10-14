#!/bin/sh
set -eu
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
python3 download_datasets.py "$@"
