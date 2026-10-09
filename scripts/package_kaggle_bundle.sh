#!/usr/bin/env bash
# Zip src/ for uploading as a Kaggle dataset (companion to milestone1_kaggle.ipynb)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/kaggle/f1-podium-src.zip"
mkdir -p "$ROOT/kaggle"
rm -f "$OUT"
(cd "$ROOT" && zip -r "$OUT" src -x '*/__pycache__/*' '*.pyc')
echo "Created $OUT ($(du -h "$OUT" | cut -f1))"
