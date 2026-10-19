"""Primary / foreign key audits for Ergast tables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class KeyAuditResult:
    table: str
    key_cols: List[str]
    duplicate_rows: int
    null_key_rows: int


@dataclass
class FkAuditResult:
    child_table: str
    child_col: str
    parent_table: str
    parent_col: str
    orphan_rows: int
    null_rows: int


def audit_primary_keys(tables: Dict[str, pd.DataFrame]) -> List[KeyAuditResult]:
    specs = {
        "circuits": ["circuitId"],
        "constructors": ["constructorId"],
        "constructor_results": ["constructorResultsId"],
        "constructor_standings": ["constructorStandingsId"],
        "drivers": ["driverId"],
        "driver_standings": ["driverStandingsId"],
        "races": ["raceId"],
        "results": ["resultId"],
        "qualifying": ["qualifyId"],
        "lap_times": ["raceId", "driverId", "lap"],
        "pit_stops": ["raceId", "driverId", "stop"],
        "sprint_results": ["resultId"],
        "status": ["statusId"],
        "seasons": ["year"],
    }
    out: List[KeyAuditResult] = []
    for table, cols in specs.items():
        if table not in tables:
            continue
        df = tables[table]
        if not set(cols).issubset(df.columns):
            # Some source revisions omit synthetic IDs; use the natural composite key.
            fallback = {
                "constructor_results": ["raceId", "constructorId"],
                "constructor_standings": ["raceId", "constructorId"],
                "driver_standings": ["raceId", "driverId"],
                "qualifying": ["raceId", "driverId"],
                "sprint_results": ["raceId", "driverId"],
            }.get(table)
            if not fallback or not set(fallback).issubset(df.columns):
                continue
            cols = fallback
        dup = int(df.duplicated(subset=cols).sum())
        nulls = int(df[cols].isna().any(axis=1).sum())
        out.append(KeyAuditResult(table, cols, dup, nulls))
    return out


def audit_foreign_keys(tables: Dict[str, pd.DataFrame]) -> List[FkAuditResult]:
    specs = [
        ("results", "raceId", "races", "raceId"),
        ("results", "driverId", "drivers", "driverId"),
        ("results", "constructorId", "constructors", "constructorId"),
        ("results", "statusId", "status", "statusId"),
        ("constructor_results", "raceId", "races", "raceId"),
        ("constructor_results", "constructorId", "constructors", "constructorId"),
        ("constructor_standings", "raceId", "races", "raceId"),
        ("constructor_standings", "constructorId", "constructors", "constructorId"),
        ("driver_standings", "raceId", "races", "raceId"),
        ("driver_standings", "driverId", "drivers", "driverId"),
        ("races", "circuitId", "circuits", "circuitId"),
        ("qualifying", "raceId", "races", "raceId"),
        ("qualifying", "driverId", "drivers", "driverId"),
        ("qualifying", "constructorId", "constructors", "constructorId"),
        ("lap_times", "raceId", "races", "raceId"),
        ("lap_times", "driverId", "drivers", "driverId"),
        ("pit_stops", "raceId", "races", "raceId"),
        ("pit_stops", "driverId", "drivers", "driverId"),
        ("sprint_results", "raceId", "races", "raceId"),
        ("sprint_results", "driverId", "drivers", "driverId"),
        ("sprint_results", "constructorId", "constructors", "constructorId"),
        ("sprint_results", "statusId", "status", "statusId"),
        ("races", "year", "seasons", "year"),
    ]
    out: List[FkAuditResult] = []
    for child_t, child_c, parent_t, parent_c in specs:
        if child_t not in tables or parent_t not in tables:
            continue
        child = tables[child_t]
        parent = tables[parent_t]
        if child_c not in child.columns or parent_c not in parent.columns:
            continue
        null_rows = int(child[child_c].isna().sum())
        orphans = child.loc[child[child_c].notna() & ~child[child_c].isin(parent[parent_c]), child_c]
        out.append(FkAuditResult(child_t, child_c, parent_t, parent_c, int(len(orphans)), null_rows))
    return out
