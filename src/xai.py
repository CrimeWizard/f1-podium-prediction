"""Global / local explainability helpers (optional shap, lime)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_table(model_pipeline, X, y, *, n_repeats: int = 5, random_state: int = 42):
    result = permutation_importance(
        model_pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)


def shap_summary(model_pipeline, X_sample: pd.DataFrame):
    import shap

    prep = model_pipeline.named_steps["prep"]
    clf = model_pipeline.named_steps["clf"]
    Xt = prep.transform(X_sample)
    explainer = shap.Explainer(clf, Xt, feature_names=_transformed_names(prep, X_sample))
    return explainer(Xt)


def _transformed_names(prep, X: pd.DataFrame):
    try:
        return prep.get_feature_names_out()
    except Exception:
        return [f"f{i}" for i in range(prep.transform(X).shape[1])]


def lime_explain_row(model_pipeline, X_row: pd.DataFrame, training_data: pd.DataFrame):
    from lime.lime_tabular import LimeTabularExplainer

    prep = model_pipeline.named_steps["prep"]
    clf = model_pipeline.named_steps["clf"]
    Xt_train = prep.transform(training_data)
    Xt_row = prep.transform(X_row)

    explainer = LimeTabularExplainer(
        Xt_train,
        mode="classification",
        feature_names=_transformed_names(prep, training_data),
        class_names=["no_podium", "podium"],
        discretize_continuous=True,
    )

    def predict_proba_fn(x):
        return clf.predict_proba(x)

    return explainer.explain_instance(
        Xt_row[0],
        predict_proba_fn,
        num_features=10,
    )
