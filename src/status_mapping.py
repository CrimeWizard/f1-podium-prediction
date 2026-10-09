"""Map Ergast status strings to retirement categories (Q3 mechanical rate)."""
from __future__ import annotations

import re
from typing import Literal

import pandas as pd

from src.de_analysis import MECHANICAL_STATUS_KEYWORDS

Category = Literal["finished", "mechanical", "accident_or_driver", "regulatory", "other_dnf"]

FINISHED_RE = re.compile(r"^(finished|\+[0-9]+ laps?)$", re.I)

ACCIDENT_DRIVER_KEYWORDS = (
    "collision",
    "accident",
    "spun",
    "driver",
    "crash",
    "barrier",
    "safety",
    "tyre",
    "wheel",
    "driveshaft",
    "driveshaft",
    "puncture",
    "off track",
    "track",
    "wing",
    "damage",
)

REGULATORY_KEYWORDS = (
    "disqualified",
    "excluded",
    "withdrawn",
    "not classified",
    "failed to qualify",
    "did not qualify",
    "black flag",
    "fuel pressure",
    "underweight",
    "technical",
    "plenum",
    "illegal",
)


def classify_status_label(label: str) -> Category:
    s = (label or "").strip().lower()
    if FINISHED_RE.match(s):
        return "finished"
    if any(k in s for k in REGULATORY_KEYWORDS):
        return "regulatory"
    mech_pat = "|".join(re.escape(k) for k in MECHANICAL_STATUS_KEYWORDS)
    if re.search(mech_pat, s):
        return "mechanical"
    if any(k in s for k in ACCIDENT_DRIVER_KEYWORDS):
        return "accident_or_driver"
    return "other_dnf"


def build_status_mapping(status: pd.DataFrame) -> pd.DataFrame:
    """One row per statusId with assigned category (document in report)."""
    out = status[["statusId", "status"]].copy()
    out["category"] = out["status"].map(classify_status_label)
    return out.sort_values(["category", "status"])
