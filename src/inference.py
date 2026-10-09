"""Inference helper for complete and partially missing pre-race records."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.train import TARGET, feature_columns


def predict_podium(
    model_pipeline,
    record: Dict[str, Any],
    *,
    threshold: float = 0.5,
    feature_names: list[str] | None = None,
) -> Dict[str, Any]:
    """
    Predict podium probability for one driver–race row.

    `record` may omit optional fields; the sklearn preprocessor imputes missing values.
    """
    feature_names = feature_names or [c for c in record.keys() if c != TARGET]
    row = {c: record.get(c, np.nan) for c in feature_names}
    X = pd.DataFrame([row])
    proba = float(model_pipeline.predict_proba(X)[0, 1])
    return {
        "podium_probability": proba,
        "podium_prediction": int(proba >= threshold),
        "threshold": threshold,
    }
