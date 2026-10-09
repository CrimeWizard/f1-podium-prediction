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

SETUP_CODE = """# Kaggle environment setup
import os
import subprocess
import sys
import zipfile
from pathlib import Path

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
INPUT = Path("/kaggle/input") if Path("/kaggle/input").exists() else None


def find_project_root() -> Path | None:
    candidates = [WORK, Path.cwd()]
    if INPUT:
        for zpath in INPUT.rglob("*.zip"):
            try:
                with zipfile.ZipFile(zpath) as zf:
                    if any(n.startswith("src/") for n in zf.namelist()):
                        zf.extractall(WORK)
                        print("Extracted zip:", zpath)
            except zipfile.BadZipFile:
                pass
        for cfg in INPUT.rglob("src/config.py"):
            return cfg.parent.parent
    for base in candidates:
        if (base / "src" / "config.py").exists():
            return base
        if (base.parent / "src" / "config.py").exists():
            return base.parent
    return None


ROOT = find_project_root()
if ROOT is None and Path("/kaggle/working").exists():
    repo = WORK / "f1-podium-prediction"
    if not (repo / "src" / "config.py").exists():
        print("Cloning GitHub repo (Internet must be ON)...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/CrimeWizard/f1-podium-prediction.git", str(repo)],
            check=True,
        )
    ROOT = repo

if ROOT is None or not (ROOT / "src" / "config.py").exists():
    raise FileNotFoundError(
        "Cannot find src/. Add dataset f1-podium-src OR enable Internet for git clone."
    )

sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

if Path("/kaggle/working").exists():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn", "shap", "lime"],
        check=False,
    )

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
