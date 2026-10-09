"""Project paths and time-split constants (GUC Milestone 1)."""
from __future__ import annotations

import os
from pathlib import Path


def _find_kaggle_csv_dir() -> Path | None:
    """Locate Ergast CSV folder under /kaggle/input when running on Kaggle."""
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None
    for child in kaggle_input.iterdir():
        if child.is_dir() and (child / "results.csv").exists():
            return child
    return None


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = project_root()
DATA_RAW = Path(os.environ.get("F1_DATA_RAW", _find_kaggle_csv_dir() or (PROJECT_ROOT / "data" / "raw")))
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
