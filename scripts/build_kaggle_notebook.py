#!/usr/bin/env python3
"""Generate notebooks/milestone1_kaggle.ipynb = Kaggle setup + local milestone1 cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_NB = ROOT / "notebooks" / "milestone1.ipynb"
OUT_NB = ROOT / "notebooks" / "milestone1_kaggle.ipynb"

SETUP_MD = """# Milestone 1 — Kaggle edition

### Required Kaggle input
- **[Formula 1 World Championship](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)** (the attached revision includes 1950–2024)
- The source code is cloned from the public GitHub repository in the setup cell; a separate `f1-podium-src` dataset is not needed.

### Settings
- Turn **Internet ON** (gear icon) for the GitHub clone and optional `shap` / `lime` installs.

### After Run All
Download outputs from the **Output** tab (`figures/`, `processed/*.csv`).
"""

SETUP_CODE = """# Kaggle environment setup (run first; Internet ON for GitHub sync)
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

# --- 2) Code lives in /kaggle/working (always sync latest main to avoid stale Kaggle code) ---
ROOT = WORK / "f1-podium-prediction"
if (ROOT / ".git").is_dir():
    subprocess.run(["git", "-C", str(ROOT), "fetch", "origin", "main"], check=True)
    subprocess.run(["git", "-C", str(ROOT), "reset", "--hard", "origin/main"], check=True)
else:
    if ROOT.exists():
        import shutil
        shutil.rmtree(ROOT)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/CrimeWizard/f1-podium-prediction.git", str(ROOT)],
        check=True,
    )

# Run All can reuse a live kernel; discard imported project modules after syncing.
for module_name in list(sys.modules):
    if module_name == "src" or module_name.startswith("src."):
        del sys.modules[module_name]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn", "shap", "lime"],
    check=False,
)

from src.config import get_data_raw

print("Project root (code):", ROOT)
print("Source revision:", subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip())
print("CSV folder (data):", get_data_raw(), "| OK:", (get_data_raw() / "results.csv").exists())
"""

IMPORTS_CODE = """# Project imports (run after the setup cell)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from pathlib import Path

ROOT = Path.cwd()
from src.audit import audit_foreign_keys, audit_primary_keys
from src.cleaning import build_base_results_table, exclude_indianapolis_500
from src.config import DATA_PROCESSED, FIGURES_DIR
from src.de_analysis import (
    question1_front_row_podium_by_circuit,
    question2_home_podium_controlled_grid,
    question3_mechanical_retirement_by_constructor,
)
from src.experiments import (
    ablation_table,
    imputation_strategy_experiment,
    indy_500_exclusion_experiment,
    preprocessing_order_experiments,
)
from src.features import build_modeling_table
from src.inference import predict_podium
from src.io import load_raw_tables, missing_tables, save_processed
from src.splits import temporal_split
from src.status_mapping import build_status_mapping
from src.train import (
    TARGET,
    collect_ffnn_metrics,
    collect_sklearn_metrics,
    feature_columns,
    train_logistic_regression,
    compare_random_forest_overfitting,
    train_random_forest,
    train_shallow_ffnn,
    pick_threshold_on_val,
)
from src.xai import permutation_importance_table

sns.set_theme(style="whitegrid")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
"""


def main() -> None:
    if not SRC_NB.exists():
        raise SystemExit("Run scripts/build_notebook.py first.")

    base = json.loads(SRC_NB.read_text())
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": SETUP_MD.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": SETUP_CODE.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": IMPORTS_CODE.splitlines(keepends=True)},
    ]
    # Skip duplicate title markdown from local notebook
    for cell in base["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell.get("source", []))
            if src.startswith("# Kaggle auto-setup (safe locally)"):
                continue
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
