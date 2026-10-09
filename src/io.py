"""Load raw Ergast CSV tables from data/raw (immutable originals)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import TABLE_FILES, get_data_raw

NA_VALUES = ["\\N", "NA", ""]


def raw_table_paths(raw_dir: Path | None = None) -> Dict[str, Path]:
    root = raw_dir or get_data_raw()
    return {name.replace(".csv", ""): root / name for name in TABLE_FILES}


def missing_tables(raw_dir: Path | None = None) -> list[str]:
    from src.kaggle_bootstrap import bootstrap_kaggle

    bootstrap_kaggle()
    paths = raw_table_paths(raw_dir)
    return [k for k, p in paths.items() if not p.exists()]


def load_raw_tables(raw_dir: Path | None = None) -> Dict[str, pd.DataFrame]:
    from src.kaggle_bootstrap import bootstrap_kaggle

    bootstrap_kaggle()
    root = raw_dir or get_data_raw()
    missing = missing_tables(root)
    if missing:
        hint = (
            "On Kaggle: Add data → Formula 1 World Championship (CSV dataset). "
            "Then enable Internet and re-run from the top."
            if Path("/kaggle/input").exists()
            else "Download CSVs into data/raw (see README)."
        )
        raise FileNotFoundError(
            f"Missing tables under {root}: " + ", ".join(missing) + ". " + hint
        )
    tables: Dict[str, pd.DataFrame] = {}
    for name, path in raw_table_paths(root).items():
        tables[name] = pd.read_csv(path, na_values=NA_VALUES, low_memory=False)
    return tables


def save_processed(df: pd.DataFrame, name: str) -> Path:
    from src.config import DATA_PROCESSED

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return out
