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


def audit_primary_keys(tables: Dict[str, pd.DataFrame]) -> List[KeyAuditResult]:
    specs = {
        "circuits": ["circuitId"],
        "constructors": ["constructorId"],
        "drivers": ["driverId"],
        "races": ["raceId"],
        "results": ["resultId"],
        "status": ["statusId"],
        "seasons": ["year"],
    }
    out: List[KeyAuditResult] = []
    for table, cols in specs.items():
        if table not in tables:
            continue
        df = tables[table]
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
        ("races", "circuitId", "circuits", "circuitId"),
        ("qualifying", "raceId", "races", "raceId"),
        ("driver_standings", "raceId", "races", "raceId"),
        ("constructor_standings", "raceId", "races", "raceId"),
    ]
    out: List[FkAuditResult] = []
    for child_t, child_c, parent_t, parent_c in specs:
        if child_t not in tables or parent_t not in tables:
            continue
        child = tables[child_t]
        parent = tables[parent_t]
        if child_c not in child.columns or parent_c not in parent.columns:
            continue
        orphans = child.loc[~child[child_c].isin(parent[parent_c]), child_c]
        out.append(FkAuditResult(child_t, child_c, parent_t, parent_c, int(orphans.notna().sum())))
    return out
