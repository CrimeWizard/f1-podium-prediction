"""Temporal train / validation / test splits."""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from src.config import TEST_YEAR_MIN, TRAIN_YEAR_MAX, VAL_YEAR_MAX, VAL_YEAR_MIN


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] <= TRAIN_YEAR_MAX].copy()
    val = df[(df["year"] >= VAL_YEAR_MIN) & (df["year"] <= VAL_YEAR_MAX)].copy()
    test = df[df["year"] >= TEST_YEAR_MIN].copy()
    return train, val, test
