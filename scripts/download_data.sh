#!/usr/bin/env bash
# Run on Wi‑Fi only. Requires Kaggle API credentials in ~/.kaggle/kaggle.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/raw"
mkdir -p "$DEST"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Install Kaggle CLI: pip install kaggle"
  exit 1
fi

echo "Downloading Ergast F1 dataset into $DEST ..."
kaggle datasets download -d rohanrao/formula-1-world-championship-1950-2020 -p "$DEST" --unzip
echo "Done. CSV count: $(ls -1 "$DEST"/*.csv 2>/dev/null | wc -l)"
