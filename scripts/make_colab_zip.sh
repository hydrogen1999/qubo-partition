#!/usr/bin/env bash
# Package the project into qubo-partition.zip for uploading to Google Colab.
# (Excludes the venv, caches, results, and any downloaded datasets.)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="qubo-partition.zip"
rm -f "$OUT"
zip -r -q "$OUT" \
    src experiments tests scripts pyproject.toml requirements.txt README.md docs \
    -x '*/__pycache__/*' '*.pyc' '.venv/*' 'results/*' 'datasets/*' '.git/*' '*.DS_Store'
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1)). Upload it in the Colab 'Get the code' cell."
