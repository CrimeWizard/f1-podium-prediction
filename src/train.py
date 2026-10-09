"""Model training & metrics (sklearn + optional Keras FFNN)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


TARGET = "podium"
ID_COLS = {"raceId", "driverId", "constructorId", "circuitId", "year"}


@dataclass
class MetricRow:
    model: str
    split: str
    roc_auc: float
    pr_auc: float
    f1: float


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ID_COLS and c != TARGET]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                cat_cols,
            ),
        ]
    )


def evaluate_probs(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Tuple[float, float, float]:
    roc = roc_auc_score(y_true, proba)
    pr = average_precision_score(y_true, proba)
    f1 = f1_score(y_true, (proba >= threshold).astype(int))
    return roc, pr, f1


def pick_threshold_on_val(y_val: np.ndarray, proba_val: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    if len(thr) == 0:
        return 0.5
    idx = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
    return float(thr[max(0, min(idx, len(thr) - 1))])


def train_logistic_regression(X_train, y_train, X_val, y_val) -> Tuple[Pipeline, float]:
    pipe = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    proba_val = pipe.predict_proba(X_val)[:, 1]
    thr = pick_threshold_on_val(y_val, proba_val)
    return pipe, thr


def train_random_forest(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_shallow_ffnn(X_train, y_train, X_val, y_val):
    """Returns fitted Keras model + preprocessor + threshold. Requires tensorflow."""
    from tensorflow import keras

    prep = build_preprocessor(X_train)
    Xtr = prep.fit_transform(X_train)
    Xva = prep.transform(X_val)
    model = keras.Sequential([
        keras.layers.Input(shape=(Xtr.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy")
    model.fit(Xtr, y_train, validation_data=(Xva, y_val), epochs=30, batch_size=256, verbose=0)
    proba_val = model.predict(Xva, verbose=0).ravel()
    thr = pick_threshold_on_val(y_val, proba_val)
    return model, prep, thr


def score_all_splits(name: str, y, proba, split_name: str, threshold: float) -> MetricRow:
    roc, pr, _ = evaluate_probs(y, proba, threshold)
    f1 = f1_score(y, (proba >= threshold).astype(int))
    return MetricRow(name, split_name, roc, pr, f1)
