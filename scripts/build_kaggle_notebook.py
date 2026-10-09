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
2. **`f1-podium-src`** — upload `kaggle/f1-podium-src.zip` **or** the whole `src/` folder as a dataset (Add data → your `f1-podium-src`)

### Settings
- Turn **Internet ON** (gear icon) for `pip install shap lime`.

### After Run All
Download outputs from the **Output** tab (`figures/`, `processed/*.csv`).
"""

SETUP_CODE = """# Kaggle environment setup (run first; Internet ON for git clone)
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
INPUT = Path("/kaggle/input") if Path("/kaggle/input").exists() else None

# --- 1) F1 CSVs live in a DIFFERENT dataset (Formula 1 World Championship) ---
csv_dir = None
if INPUT:
    for results in INPUT.rglob("results.csv"):
        folder = results.parent
        if (folder / "circuits.csv").exists() and (folder / "races.csv").exists():
            csv_dir = folder
            break
if csv_dir is None:
    raise FileNotFoundError(
        "Add the Formula 1 World Championship CSV dataset (rohanrao or Vopani) in Add data."
    )
os.environ["F1_DATA_RAW"] = str(csv_dir)

# --- 2) Code lives in /kaggle/working (clone full repo; do not use f1-podium-src as ROOT) ---
ROOT = WORK / "f1-podium-prediction"
if not (ROOT / "src" / "config.py").exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/CrimeWizard/f1-podium-prediction.git", str(ROOT)],
        check=True,
    )

sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn", "shap", "lime"],
    check=False,
)

from src.config import get_data_raw

print("Project root (code):", ROOT)
print("CSV folder (data):", get_data_raw(), "| OK:", (get_data_raw() / "results.csv").exists())
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
