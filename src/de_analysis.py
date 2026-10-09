"""Data-engineering questions 1–3 (multi-hop joins)."""
from __future__ import annotations

import re

import pandas as pd

from src.config import GROUND_EFFECT_START, HYBRID_ERA_START, HYBRID_ERA_END

# Mechanical retirement status labels (document & extend in notebook EDA)
MECHANICAL_STATUS_KEYWORDS = (
    "engine",
    "gearbox",
    "hydraul",
    "brake",
    "power unit",
    "electrical",
    "suspension",
    "steering",
    "throttle",
    "clutch",
    "driveshaft",
    "exhaust",
    "fuel",
    "oil",
    "water",
    "radiator",
    "turbo",
    "ers",
    "battery",
    "mgu",
    "overheating",
    "vibrations",
    "wheel",
    "tyre",
    "tire",
)


def _status_is_mechanical(status_series: pd.Series) -> pd.Series:
    s = status_series.fillna("").str.lower()
    pattern = "|".join(re.escape(k) for k in MECHANICAL_STATUS_KEYWORDS)
    return s.str.contains(pattern, regex=True)


def question1_front_row_podium_by_circuit(
    base: pd.DataFrame,
    circuits: pd.DataFrame,
    *,
    use_grid: bool = True,
) -> pd.DataFrame:
    """
    Q1: front-row (grid 1–2) → podium rate by circuit; split pre/post hybrid era.
    use_grid=True uses actual grid; False uses qualifying position proxy if available.
    """
    df = base.copy()
    pos_col = "grid" if use_grid else "grid"
    front = df[df[pos_col].isin([1, 2])].copy()
    front["era"] = pd.cut(
        front["year"],
        bins=[-float("inf"), HYBRID_ERA_START - 1, float("inf")],
        labels=[f"pre_{HYBRID_ERA_START}", f"from_{HYBRID_ERA_START}"],
    )
    agg = (
        front.groupby(["circuitId", "era"], observed=True)
        .agg(front_row_starts=("podium", "count"), podiums=("podium", "sum"))
        .reset_index()
    )
    agg["podium_rate"] = agg["podiums"] / agg["front_row_starts"]
    return agg.merge(circuits[["circuitId", "name", "country"]], on="circuitId", how="left")


def _normalize_country(name: str) -> str:
    mapping = {
        "uk": "united kingdom",
        "great britain": "united kingdom",
        "usa": "united states",
        "uae": "united arab emirates",
    }
    if not isinstance(name, str):
        return ""
    n = name.strip().lower()
    return mapping.get(n, n)


def question2_home_podium_controlled_grid(
    base: pd.DataFrame,
    drivers: pd.DataFrame,
    circuits: pd.DataFrame,
    *,
    grid_bins: list[int] | None = None,
) -> pd.DataFrame:
    """
    Q2: podium rate home vs away within grid-position bins.
    """
    grid_bins = grid_bins or [0, 5, 10, 15, 25]
    df = base.merge(drivers[["driverId", "nationality"]], on="driverId", how="left")
    df = df.merge(circuits[["circuitId", "country"]], on="circuitId", how="left", suffixes=("_driver", "_circuit"))
    df["home_race"] = (
        df["nationality"].map(_normalize_country) == df["country"].map(_normalize_country)
    ).astype(int)
    df["grid_bin"] = pd.cut(df["grid"].fillna(99), bins=grid_bins, right=True)
    agg = (
        df.groupby(["grid_bin", "home_race"], observed=True)
        .agg(starts=("podium", "count"), podiums=("podium", "sum"))
        .reset_index()
    )
    agg["podium_rate"] = agg["podiums"] / agg["starts"]
    return agg


def question3_mechanical_retirement_by_constructor(
    base: pd.DataFrame,
    status: pd.DataFrame,
    constructors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Q3: mechanical retirement rate by constructor, 2014–2021 vs 2022–2024.
    """
    from src.status_mapping import build_status_mapping

    mapping = build_status_mapping(status)
    st = mapping.rename(columns={"status": "status_text"})
    df = base.merge(st, on="statusId", how="left")
    df["mechanical_dnf"] = df["category"] == "mechanical"

    def era_label(y: int) -> str | None:
        if HYBRID_ERA_START <= y <= HYBRID_ERA_END:
            return f"{HYBRID_ERA_START}_{HYBRID_ERA_END}"
        if y >= GROUND_EFFECT_START:
            return f"{GROUND_EFFECT_START}_plus"
        return None

    df["era"] = df["year"].map(era_label)
    df = df[df["era"].notna() & (df["grid"].fillna(0) > 0)]

    agg = (
        df.groupby(["constructorId", "era"])
        .agg(starts=("resultId", "count"), mechanical_retirements=("mechanical_dnf", "sum"))
        .reset_index()
    )
    agg["mech_rate"] = agg["mechanical_retirements"] / agg["starts"]
    return agg.merge(constructors[["constructorId", "name"]], on="constructorId", how="left")
