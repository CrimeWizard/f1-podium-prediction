"""Project paths and time-split constants (GUC Milestone 1)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

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
