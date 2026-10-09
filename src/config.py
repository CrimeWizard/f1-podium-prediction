"""Project paths and time-split constants (GUC Milestone 1)."""
from __future__ import annotations

import os
from pathlib import Path


def _find_kaggle_csv_dir() -> Path | None:
    """Locate Ergast CSV folder under /kaggle/input (nested dataset paths included)."""
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None
    for results in kaggle_input.rglob("results.csv"):
        parent = results.parent
        # Skip code-only datasets that might ship a stub csv
        if (parent / "circuits.csv").exists() and (parent / "races.csv").exists():
            return parent
    return None


def get_data_raw() -> Path:
    """Resolve CSV directory (Kaggle input, env override, or local data/raw)."""
    explicit = os.environ.get("F1_DATA_RAW")
    if explicit:
        return Path(explicit)
    found = _find_kaggle_csv_dir()
    if found:
        return found
    return project_root() / "data" / "raw"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = project_root()
# Prefer get_data_raw() — DATA_RAW is evaluated at import time and may be stale on Kaggle.
DATA_RAW = get_data_raw()
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# On Kaggle, write artifacts under /kaggle/working
if Path("/kaggle/working").exists():
    _work = Path("/kaggle/working")
    DATA_PROCESSED = _work / "processed"
    FIGURES_DIR = _work / "figures"
    MODELS_DIR = _work / "models"

# Suggested temporal split (documented in newcomer PDF)
TRAIN_YEAR_MAX = 2019
VAL_YEAR_MIN = 2020
VAL_YEAR_MAX = 2021
TEST_YEAR_MIN = 2022

HYBRID_ERA_START = 2014
HYBRID_ERA_END = 2021
GROUND_EFFECT_START = 2022

# Indianapolis 500 was a championship round 1950–1960 (optional exclusion for Q2)
INDY_500_RACE_NAME = "Indianapolis 500"

KAGGLE_DATASET = "rohanrao/formula-1-world-championship-1950-2020"

TABLE_FILES = [
    "circuits.csv",
    "constructors.csv",
    "constructor_results.csv",
    "constructor_standings.csv",
    "drivers.csv",
    "driver_standings.csv",
    "races.csv",
    "results.csv",
    "qualifying.csv",
    "lap_times.csv",
    "pit_stops.csv",
    "sprint_results.csv",
    "seasons.csv",
    "status.csv",
]
