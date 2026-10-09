#!/usr/bin/env python3
"""Generate notebooks/milestone1_kaggle.ipynb = Kaggle setup + local milestone1 cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NB = ROOT / "notebooks" / "milestone1.ipynb"
OUT_NB = ROOT / "notebooks" / "milestone1_kaggle.ipynb"

SETUP_MD = """# Milestone 1 — Kaggle edition

### Add these datasets (Notebook → Add data)
1. **[Formula 1 World Championship 1950-2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)** (required)
2. **`f1-podium-src.zip`** — run `bash scripts/package_kaggle_bundle.sh` locally, upload zip as a new Kaggle dataset named e.g. `f1-podium-src`

### Settings
- Turn **Internet ON** (gear icon) for `pip install shap lime`.

### After Run All
Download outputs from the **Output** tab (`figures/`, `processed/*.csv`).
"""

SETUP_CODE = """# Kaggle environment setup
import os
import sys
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()

for zpath in Path("/kaggle/input").rglob("f1-podium-src.zip") if Path("/kaggle/input").exists() else []:
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(WORK)
    print("Extracted", zpath)

ROOT = WORK if (WORK / "src").exists() else Path.cwd()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

if Path("/kaggle/working").exists():
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn", "shap", "lime"], check=False)

from src.config import DATA_RAW
print("Project root:", ROOT)
print("CSV folder:", DATA_RAW, "| OK:", (DATA_RAW / "results.csv").exists())
"""


def main() -> None:
    if not SRC_NB.exists():
        raise SystemExit("Run scripts/build_notebook.py first.")

    base = json.loads(SRC_NB.read_text())
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": SETUP_MD.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": SETUP_CODE.splitlines(keepends=True)},
    ]
    # Skip duplicate title markdown from local notebook
    for cell in base["cells"]:
        if cell["cell_type"] == "markdown":
            src = "".join(cell.get("source", []))
            if src.startswith("# Milestone 1: F1 Podium Prediction"):
                continue
        cells.append(cell)

    out = {**base, "cells": cells}
    OUT_NB.write_text(json.dumps(out, indent=1))
    print("Wrote", OUT_NB)


if __name__ == "__main__":
    main()
