#!/usr/bin/env python3
"""Generate notebooks/milestone1.ipynb (no third-party deps)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "milestone1.ipynb"

cells = []

def md(source: str):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})

def code(source: str):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": source.splitlines(keepends=True)})

md("""# Milestone 1: F1 Podium Prediction

**Target:** `podium` = `positionOrder <= 3`  
**Cutoff:** features only from after qualifying / before race start.

Run this notebook top-to-bottom after placing Ergast CSVs in `data/raw/`.
""")

code("""import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.audit import audit_foreign_keys, audit_primary_keys
from src.cleaning import build_base_results_table
from src.config import FIGURES_DIR, DATA_PROCESSED
from src.de_analysis import (
    question1_front_row_podium_by_circuit,
    question2_home_podium_controlled_grid,
    question3_mechanical_retirement_by_constructor,
)
from src.experiments import ablation_table, preprocessing_order_experiments
from src.features import build_modeling_table
from src.inference import predict_podium
from src.io import load_raw_tables, missing_tables, save_processed
from src.splits import temporal_split
from src.train import (
    TARGET,
    feature_columns,
    score_all_splits,
    train_logistic_regression,
    train_random_forest,
    train_shallow_ffnn,
)
from src.xai import permutation_importance_table

sns.set_theme(style="whitegrid")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
""")

md("## 0. Load data")
code("""missing = missing_tables()
if missing:
    raise FileNotFoundError(
        "Place Kaggle CSV files in data/raw. Missing: " + ", ".join(missing)
    )

tables = load_raw_tables()
list(tables.keys())
""")

md("## 1. EDA (before cleaning)")
code("""results_raw = tables["results"]
fig, ax = plt.subplots(figsize=(8, 4))
results_raw["positionOrder"].hist(bins=30, ax=ax)
ax.set_title("positionOrder distribution (raw)")
plt.savefig(FIGURES_DIR / "eda_position_order_raw.png", bbox_inches="tight")
plt.show()

qual = tables["qualifying"]
coverage = qual.groupby("raceId").size().rename("rows")
races = tables["races"][["raceId", "year"]]
cov = races.merge(coverage, on="raceId", how="left").fillna(0)
cov_by_year = cov.groupby("year")["rows"].mean()
cov_by_year.plot(figsize=(9, 4), title="Mean qualifying rows per race by season")
plt.savefig(FIGURES_DIR / "eda_qualifying_coverage_by_year.png", bbox_inches="tight")
plt.show()
""")

md("## 2. Auditing keys & cleaning")
code("""pk = pd.DataFrame([r.__dict__ for r in audit_primary_keys(tables)])
fk = pd.DataFrame([r.__dict__ for r in audit_foreign_keys(tables)])
display(pk)
display(fk)

base, duplicate_audit = build_base_results_table(
    tables,
    exclude_indy=True,
    dedupe_policy="best_position_order",
)
print("Duplicate (raceId, driverId) rows before policy:", len(duplicate_audit))
assert base.duplicated(["raceId", "driverId"]).sum() == 0
save_processed(base, "base_results")
""")

md("## 3. EDA (after cleaning)")
code("""fig, ax = plt.subplots(figsize=(6, 4))
base.groupby("year")["podium"].mean().plot(ax=ax)
ax.set_title("Podium rate by season (after cleaning)")
plt.savefig(FIGURES_DIR / "eda_podium_rate_by_year.png", bbox_inches="tight")
plt.show()
""")

md("## 4. Data-engineering questions")
code("""q1 = question1_front_row_podium_by_circuit(base, tables["circuits"])
q1_top = q1.sort_values("podium_rate", ascending=False).groupby("era").head(10)
display(q1_top)

plt.figure(figsize=(10, 5))
for era, g in q1.groupby("era"):
    g.nlargest(15, "podium_rate").plot(x="name", y="podium_rate", kind="bar", title=f"Q1 top circuits {era}")
plt.savefig(FIGURES_DIR / "de_q1_front_row_podium.png", bbox_inches="tight")
plt.show()

q2 = question2_home_podium_controlled_grid(base, tables["drivers"], tables["circuits"])
display(q2)
q2.pivot(index="grid_bin", columns="home_race", values="podium_rate").plot(kind="bar", figsize=(8, 4), title="Q2 home vs away by grid bin")
plt.savefig(FIGURES_DIR / "de_q2_home_advantage.png", bbox_inches="tight")
plt.show()

q3 = question3_mechanical_retirement_by_constructor(base, tables["status"], tables["constructors"])
display(q3.sort_values("mech_rate", ascending=False).groupby("era").head(10))
""")

md("## 5. Feature engineering (leakage-safe)")
code("""model_df = build_modeling_table(tables, base)
save_processed(model_df, "modeling_table")
model_df.head()
""")

md("## 6. Train / validation / test split")
code("""train_df, val_df, test_df = temporal_split(model_df)
len(train_df), len(val_df), len(test_df)
""")

md("## 7. Preprocessing-order experiments")
code("""prep_cmp = preprocessing_order_experiments(train_df, val_df)
display(prep_cmp)
""")

md("## 8. Modeling")
code("""feats = feature_columns(train_df)
X_train, y_train = train_df[feats], train_df[TARGET]
X_val, y_val = val_df[feats], val_df[TARGET]
X_test, y_test = test_df[feats], test_df[TARGET]

lr, thr = train_logistic_regression(X_train, y_train, X_val, y_val)
rf = train_random_forest(X_train, y_train)

metrics = []
for name, model in [("logistic_regression", lr), ("random_forest", rf)]:
    for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        proba = model.predict_proba(X)[:, 1]
        metrics.append(score_all_splits(name, y, proba, split_name, thr if name == "logistic_regression" else 0.5))

try:
    nn, nn_prep, nn_thr = train_shallow_ffnn(X_train, y_train, X_val, y_val)
    Xt = nn_prep.transform(X_test)
    proba = nn.predict(Xt, verbose=0).ravel()
    metrics.append(score_all_splits("shallow_ffnn", y_test, proba, "test", nn_thr))
except Exception as e:
    print("FFNN skipped (install tensorflow):", e)

metrics_df = pd.DataFrame([m.__dict__ for m in metrics])
display(metrics_df)
""")

md("## 9. Feature-group ablation")
code("""abl = ablation_table(train_df, val_df, test_df)
display(abl)
""")

md("## 10. Explainability (XAI)")
code("""# Global: permutation importance on validation set
imp = permutation_importance_table(lr, X_val, y_val)
display(imp.head(15))

try:
    from src.xai import shap_summary
    shap_values = shap_summary(lr, X_val.sample(min(500, len(X_val)), random_state=42))
    import shap
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.savefig(FIGURES_DIR / "xai_shap_beeswarm.png", bbox_inches="tight")
    plt.show()
except Exception as e:
    print("SHAP skipped:", e)

try:
    from src.xai import lime_explain_row
    row = X_val.iloc[[0]]
    exp = lime_explain_row(lr, row, X_train.sample(min(2000, len(X_train)), random_state=0))
    exp.as_pyplot_figure()
    plt.savefig(FIGURES_DIR / "xai_lime_row0.png", bbox_inches="tight")
    plt.show()
except Exception as e:
    print("LIME skipped:", e)
""")

md("""
**Interpretation note:** Important features show what the model uses to predict — not proof they *cause* podiums.
""")

md("## 11. Inference function demo")
code("""complete = X_val.iloc[0].to_dict()
partial = {k: complete[k] for k in ["grid", "year"]}
print("Complete row:", predict_podium(lr, complete, threshold=thr, feature_names=feats))
print("Partial row:", predict_podium(lr, partial, threshold=thr, feature_names=feats))
""")

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

NB.parent.mkdir(parents=True, exist_ok=True)
NB.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {NB}")
