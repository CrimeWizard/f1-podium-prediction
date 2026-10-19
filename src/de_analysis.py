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

NON_START_STATUS_PATTERN = (
    r"did not start|not started|withdrew|withdrawn|did not qualify|failed to qualify|"
    r"not qualified|did not prequalify|entry withdrawn|did not participate|dns"
)


def _race_starters(base: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    """Keep result rows that represent a race start, including pit-lane starts."""
    status_text = status[["statusId", "status"]].rename(columns={"status": "status_text"})
    df = base.merge(status_text, on="statusId", how="left")
    known_status = df["status_text"].notna() & df["status_text"].str.strip().ne("")
    not_started = df["status_text"].fillna("").str.lower().str.contains(
        NON_START_STATUS_PATTERN, regex=True
    )
    return df.loc[known_status & ~not_started].copy()


def _status_is_mechanical(status_series: pd.Series) -> pd.Series:
    s = status_series.fillna("").str.lower()
    pattern = "|".join(re.escape(k) for k in MECHANICAL_STATUS_KEYWORDS)
    return s.str.contains(pattern, regex=True)


def question1_front_row_podium_by_circuit(
    base: pd.DataFrame,
    circuits: pd.DataFrame,
    qualifying: pd.DataFrame,
    status: pd.DataFrame,
) -> pd.DataFrame:
    """
    Q1: compare qualifying-front-row and actual-grid-front-row podium rates.

    The two bases are reported separately because grid penalties can move a
    driver away from their qualifying position. Missing qualifying records are
    excluded from the qualifying-front-row denominator, but remain eligible in
    the grid-front-row analysis.
    """
    if "position" not in qualifying.columns:
        raise ValueError("Q1 requires the qualifying table with a 'position' column")

    starters = _race_starters(base, status)
    q_positions = qualifying[["raceId", "driverId", "position"]].rename(
        columns={"position": "qualifying_position"}
    ).copy()
    q_positions["qualifying_position"] = pd.to_numeric(q_positions["qualifying_position"], errors="coerce")
    valid_q_keys = q_positions.dropna(subset=["qualifying_position"])[["raceId", "driverId"]].drop_duplicates()
    missing_quali = len(starters.merge(valid_q_keys, on=["raceId", "driverId"], how="left", indicator=True).query("_merge == 'left_only'"))
    grid_front = starters.loc[starters["grid"].isin([1, 2])].copy()
    grid_front["start_basis"] = "actual_grid_front_row"
    quali = q_positions.dropna(subset=["qualifying_position"]).sort_values("qualifying_position").drop_duplicates(
        ["raceId", "driverId"], keep="first"
    )
    quali_front = starters.merge(quali, on=["raceId", "driverId"], how="inner")
    quali_front = quali_front.loc[quali_front["qualifying_position"].isin([1, 2])].copy()
    quali_front["start_basis"] = "qualifying_front_row"
    front = pd.concat([grid_front, quali_front], ignore_index=True)
    front["era"] = pd.cut(
        front["year"],
        bins=[-float("inf"), HYBRID_ERA_START - 1, float("inf")],
        labels=[f"pre_{HYBRID_ERA_START}", f"from_{HYBRID_ERA_START}"],
    )
    agg = (
        front.groupby(["start_basis", "circuitId", "era"], observed=True)
        .agg(front_row_starts=("podium", "count"), podiums=("podium", "sum"))
        .reset_index()
    )
    agg["podium_rate"] = agg["podiums"] / agg["front_row_starts"]
    out = agg.merge(circuits[["circuitId", "name", "country"]], on="circuitId", how="left")
    out.attrs["nonstarters_excluded"] = int(len(base) - len(starters))
    out.attrs["starters_without_qualifying_position"] = int(missing_quali)
    return out


def _normalize_country(name: str) -> str:
    mapping = {
        "uk": "united kingdom",
        "great britain": "united kingdom",
        "usa": "united states",
        "uae": "united arab emirates",
        # Ergast driver nationalities are demonyms while circuit countries are
        # country names. Normalize common demonyms before comparing them.
        "british": "united kingdom",
        "english": "united kingdom",
        "scottish": "united kingdom",
        "welsh": "united kingdom",
        "american": "united states",
        "dutch": "netherlands",
        "french": "france",
        "german": "germany",
        "italian": "italy",
        "spanish": "spain",
        "belgian": "belgium",
        "austrian": "austria",
        "swiss": "switzerland",
        "brazilian": "brazil",
        "argentine": "argentina",
        "argentinian": "argentina",
        "mexican": "mexico",
        "canadian": "canada",
        "japanese": "japan",
        "australian": "australia",
        "new zealander": "new zealand",
        "south african": "south africa",
        "finnish": "finland",
        "swedish": "sweden",
        "danish": "denmark",
        "monegasque": "monaco",
        "colombian": "colombia",
        "venezuelan": "venezuela",
        "polish": "poland",
        "portuguese": "portugal",
        "russian": "russia",
        "indian": "india",
        "indonesian": "indonesia",
        "thai": "thailand",
        "chilean": "chile",
        "czech": "czech republic",
        "east german": "germany",
        "hungarian": "hungary",
        "irish": "ireland",
        "liechtensteiner": "liechtenstein",
        "malaysian": "malaysia",
        "rhodesian": "zimbabwe",
        "turkish": "turkey",
        "uruguayan": "uruguay",
    }
    if not isinstance(name, str):
        return ""
    n = name.strip().lower()
    return mapping.get(n, n)


def question2_home_podium_controlled_grid(
    base: pd.DataFrame,
    drivers: pd.DataFrame,
    circuits: pd.DataFrame,
    status: pd.DataFrame,
    *,
    grid_bins: list[int] | None = None,
) -> pd.DataFrame:
    """
    Q2: podium rate home vs away within grid-position bins.
    """
    default_bins = grid_bins is None
    grid_bins = grid_bins or [0, 5, 10, 15, float("inf")]
    df = _race_starters(base, status)
    df = df.merge(drivers[["driverId", "nationality"]], on="driverId", how="left")
    df = df.merge(circuits[["circuitId", "country"]], on="circuitId", how="left", suffixes=("_driver", "_circuit"))
    df["driver_country_norm"] = df["nationality"].map(_normalize_country)
    df["circuit_country_norm"] = df["country"].map(_normalize_country)
    # Unknown identity/country must not be silently treated as either home or away.
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    known_country = df["driver_country_norm"].ne("") & df["circuit_country_norm"].ne("")
    valid_grid = df["grid"].gt(0)
    unknown_country_count = int((~known_country).sum())
    invalid_grid_count = int((known_country & ~valid_grid).sum())
    df = df.loc[known_country & valid_grid].copy()
    df["home_race"] = (df["driver_country_norm"] == df["circuit_country_norm"]).astype(int)
    # Keep every valid starting position and leave missing/zero grids out.
    labels = ["1–5", "6–10", "11–15", "16+"] if default_bins else None
    df["grid_bin"] = pd.cut(df["grid"], bins=grid_bins, right=True, labels=labels)
    agg = (
        df.groupby(["grid_bin", "home_race"], observed=True)
        .agg(starts=("podium", "count"), podiums=("podium", "sum"))
        .reset_index()
    )
    agg["podium_rate"] = agg["podiums"] / agg["starts"]
    agg.attrs["unknown_country_rows_excluded"] = unknown_country_count
    agg.attrs["zero_or_missing_grid_rows_excluded"] = invalid_grid_count
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
    starters = _race_starters(base, status)
    df = starters.merge(
        mapping[["statusId", "category"]], on="statusId", how="left"
    )
    df["mechanical_dnf"] = df["category"] == "mechanical"

    def era_label(y: int) -> str | None:
        if HYBRID_ERA_START <= y <= HYBRID_ERA_END:
            return f"{HYBRID_ERA_START}_{HYBRID_ERA_END}"
        if GROUND_EFFECT_START <= y <= 2024:
            return f"{GROUND_EFFECT_START}_2024"
        return None

    df["era"] = df["year"].map(era_label)
    df = df[df["era"].notna()]

    agg = (
        df.groupby(["constructorId", "era"])
        .agg(starts=("resultId", "count"), mechanical_retirements=("mechanical_dnf", "sum"))
        .reset_index()
    )
    agg["mech_rate"] = agg["mechanical_retirements"] / agg["starts"]
    out = agg.merge(constructors[["constructorId", "name"]], on="constructorId", how="left")
    out.attrs["nonstarters_excluded"] = int(len(base) - len(starters))
    return out
