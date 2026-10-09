"""Feature-group ablation and preprocessing-order experiments."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.train import TARGET, feature_columns, train_logistic_regression, evaluate_probs


FEATURE_GROUPS: Dict[str, List[str]] = {
    "grid": ["grid"],
    "qualifying": ["quali_position", "q1_sec", "q2_sec", "q3_sec"],
    "standings": [
        "points_before",
        "position_before",
        "constructor_points_before",
        "constructor_position_before",
    ],
    "form": ["podium_rate_prev"],
}


def drop_feature_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
    cols = FEATURE_GROUPS.get(group, [])
    return df.drop(columns=[c for c in cols if c in df.columns])


def ablation_table(train_df, val_df, test_df) -> pd.DataFrame:
    rows = []
    for group in ["none"] + list(FEATURE_GROUPS.keys()):
        tr = train_df if group == "none" else drop_feature_group(train_df, group)
        va = val_df if group == "none" else drop_feature_group(val_df, group)
        te = test_df if group == "none" else drop_feature_group(test_df, group)
        feats = feature_columns(tr)
        X_train, y_train = tr[feats], tr[TARGET]
        X_val, y_val = va[feats], va[TARGET]
        X_test, y_test = te[feats], te[TARGET]
        pipe, thr = train_logistic_regression(X_train, y_train, X_val, y_val)
        proba_val = pipe.predict_proba(X_val)[:, 1]
        proba_test = pipe.predict_proba(X_test)[:, 1]
        _, pr_val, _ = evaluate_probs(y_val, proba_val, thr)
        _, pr_test, _ = evaluate_probs(y_test, proba_test, thr)
        rows.append({"feature_group_removed": group, "pr_auc_val": pr_val, "pr_auc_test": pr_test})
    return pd.DataFrame(rows)


def preprocessing_order_experiments(train_df, val_df) -> pd.DataFrame:
    """Compare impute→scale vs scale→impute (numeric columns only, logistic baseline)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feats = feature_columns(train_df)
    X_train, y_train = train_df[feats], train_df[TARGET]
    X_val, y_val = val_df[feats], val_df[TARGET]
    num_cols = X_train.select_dtypes(include="number").columns.tolist()

    configs = {
        "impute_then_scale": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "scale_then_impute": Pipeline([
            ("scaler", StandardScaler()),
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
    }
    rows = []
    for name, pipe in configs.items():
        pipe.fit(X_train[num_cols], y_train)
        proba = pipe.predict_proba(X_val[num_cols])[:, 1]
        rows.append({"preprocessing_order": name, "pr_auc_val": average_precision_score(y_val, proba)})
    return pd.DataFrame(rows)


def imputation_strategy_experiment(train_df, val_df) -> pd.DataFrame:
    """Experiment 2: median vs mean imputation (numeric columns, logistic baseline)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    feats = feature_columns(train_df)
    X_train, y_train = train_df[feats], train_df[TARGET]
    X_val, y_val = val_df[feats], val_df[TARGET]
    num_cols = X_train.select_dtypes(include="number").columns.tolist()

    rows = []
    for strategy in ("median", "mean"):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=strategy)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
        pipe.fit(X_train[num_cols], y_train)
        proba = pipe.predict_proba(X_val[num_cols])[:, 1]
        rows.append({"imputation_strategy": strategy, "pr_auc_val": average_precision_score(y_val, proba)})
    return pd.DataFrame(rows)


def indy_500_exclusion_experiment(tables) -> pd.DataFrame:
    """Compare PR-AUC with vs without Indianapolis 500 championship rounds."""
    from src.cleaning import build_base_results_table
    from src.features import build_modeling_table
    from src.splits import temporal_split

    rows = []
    for exclude_indy, label in [(True, "exclude_indy_500"), (False, "include_indy_500")]:
        base, _ = build_base_results_table(tables, exclude_indy=exclude_indy)
        model_df = build_modeling_table(tables, base)
        train_df, val_df, test_df = temporal_split(model_df)
        feats = feature_columns(train_df)
        pipe, thr = train_logistic_regression(
            train_df[feats], train_df[TARGET], val_df[feats], val_df[TARGET]
        )
        proba_val = pipe.predict_proba(val_df[feats])[:, 1]
        proba_test = pipe.predict_proba(test_df[feats])[:, 1]
        _, pr_val, _ = evaluate_probs(val_df[TARGET], proba_val, thr)
        _, pr_test, _ = evaluate_probs(test_df[TARGET], proba_test, thr)
        rows.append({
            "indy_policy": label,
            "n_rows": len(model_df),
            "pr_auc_val": pr_val,
            "pr_auc_test": pr_test,
        })
    return pd.DataFrame(rows)
