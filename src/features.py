"""Leakage-safe feature table for podium classification."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.cleaning import clean_qualifying_times


def pre_race_driver_standings(driver_standings: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Standings as known BEFORE each race (same raceId row = after previous round)."""
    ds = driver_standings.merge(races[["raceId", "year", "date"]], on="raceId", how="left")
    ds = ds.sort_values(["driverId", "date"])
    ds["points_before"] = ds.groupby("driverId")["points"].shift(1).fillna(0)
    ds["position_before"] = ds.groupby("driverId")["position"].shift(1)
    return ds[["raceId", "driverId", "points_before", "position_before"]]


def pre_race_constructor_standings(constructor_standings: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    cs = constructor_standings.merge(races[["raceId", "year", "date"]], on="raceId", how="left")
    cs = cs.sort_values(["constructorId", "date"])
    cs["points_before"] = cs.groupby("constructorId")["points"].shift(1).fillna(0)
    cs["position_before"] = cs.groupby("constructorId")["position"].shift(1)
    return cs.rename(
        columns={
            "points_before": "constructor_points_before",
            "position_before": "constructor_position_before",
        }
    )[["raceId", "constructorId", "constructor_points_before", "constructor_position_before"]]


def rolling_driver_form(base: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Podium rate in previous `window` races (excludes current race)."""
    df = base.sort_values(["driverId", "date"]).copy()
    df["podiums_prev"] = (
        df.groupby("driverId")["podium"]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
        .reset_index(level=0, drop=True)
    )
    df["starts_prev"] = (
        df.groupby("driverId")["podium"]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).count())
        .reset_index(level=0, drop=True)
    )
    df["podium_rate_prev"] = (df["podiums_prev"] / df["starts_prev"].replace(0, np.nan)).astype("float64")
    return df[["raceId", "driverId", "podium_rate_prev"]]


def build_modeling_table(tables: dict, base: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-race features only. Target `podium` kept for training; drop before fit.
    """
    races = tables["races"]
    qualifying = clean_qualifying_times(tables["qualifying"])
    if "position" in qualifying.columns:
        quali = qualifying[["raceId", "driverId", "position", "q1_sec", "q2_sec", "q3_sec"]].rename(
            columns={"position": "quali_position"}
        )
    else:
        quali = qualifying[["raceId", "driverId", "q1_sec", "q2_sec", "q3_sec"]]

    df = base.copy()
    df = df.merge(quali, on=["raceId", "driverId"], how="left")
    df = df.merge(pre_race_driver_standings(tables["driver_standings"], races), on=["raceId", "driverId"], how="left")
    df = df.merge(
        pre_race_constructor_standings(tables["constructor_standings"], races),
        on=["raceId", "constructorId"],
        how="left",
    )
    df = df.merge(rolling_driver_form(df), on=["raceId", "driverId"], how="left")

    # Allowed pre-race fields
    keep = [
        "raceId",
        "driverId",
        "constructorId",
        "circuitId",
        "year",
        "grid",
        "quali_position",
        "q1_sec",
        "q2_sec",
        "q3_sec",
        "points_before",
        "position_before",
        "constructor_points_before",
        "constructor_position_before",
        "podium_rate_prev",
        "podium",
    ]
    existing = [c for c in keep if c in df.columns]
    out = df[existing].copy()
    for col in out.select_dtypes(include="number").columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out
