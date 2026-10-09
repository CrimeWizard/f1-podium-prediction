"""Cleaning: sentinels, typing, grain (raceId, driverId), duplicate policy."""
from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import INDY_500_RACE_NAME


def lap_time_to_seconds(value) -> float | np.floating | None:
    """Convert qualifying lap time 'M:SS.mmm' or 'SS.mmm' to seconds."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "\\n"}:
        return np.nan
    if ":" in s:
        mins, rest = s.split(":", 1)
        return int(mins) * 60 + float(rest)
    return float(s)


def clean_qualifying_times(qualifying: pd.DataFrame) -> pd.DataFrame:
    q = qualifying.copy()
    for col in ("q1", "q2", "q3"):
        if col in q.columns:
            q[f"{col}_sec"] = q[col].map(lap_time_to_seconds)
    return q


def clean_races_dates(races: pd.DataFrame) -> pd.DataFrame:
    r = races.copy()
    if "date" in r.columns:
        r["date"] = pd.to_datetime(r["date"], errors="coerce")
    return r


def dedupe_driver_race_rows(
    results: pd.DataFrame,
    *,
    policy: str = "best_position_order",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enforce one row per (raceId, driverId).

    policy:
      - best_position_order: keep lowest positionOrder (best finish) — defensible when
        shared-drive rows reflect multiple stints/cars.
      - started_car: keep row with grid > 0 when available, else best positionOrder.
    """
    r = results.copy()
    dup_mask = r.duplicated(subset=["raceId", "driverId"], keep=False)
    dups = r.loc[dup_mask].sort_values(["raceId", "driverId", "positionOrder"])

    if not dup_mask.any():
        return r, dups

    def pick_group(g: pd.DataFrame) -> pd.DataFrame:
        if policy == "started_car":
            started = g[g["grid"].fillna(0) > 0]
            if len(started) == 1:
                return started.iloc[[0]]
            if len(started) > 1:
                return started.sort_values("positionOrder").iloc[[0]]
        return g.sort_values("positionOrder").iloc[[0]]

    kept = (
        r.groupby(["raceId", "driverId"], group_keys=False)
        .apply(pick_group)
        .reset_index(drop=True)
    )
    return kept, dups


def exclude_indianapolis_500(results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Optional: drop Indy 500 rounds (1950–1960) for home-race / field-size sanity."""
    indy_ids = races.loc[races["name"] == INDY_500_RACE_NAME, "raceId"]
    return results.loc[~results["raceId"].isin(indy_ids)].copy()


def build_base_results_table(
    tables: dict,
    *,
    exclude_indy: bool = True,
    dedupe_policy: str = "best_position_order",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge results with race year; clean grain; add podium target."""
    results = tables["results"].copy()
    races = clean_races_dates(tables["races"])
    if exclude_indy:
        results = exclude_indianapolis_500(results, races)

    results, duplicate_audit = dedupe_driver_race_rows(results, policy=dedupe_policy)
    results = results.merge(races[["raceId", "year", "circuitId", "date", "name"]], on="raceId", how="left")
    results["podium"] = (results["positionOrder"] <= 3).astype(int)
    return results, duplicate_audit
