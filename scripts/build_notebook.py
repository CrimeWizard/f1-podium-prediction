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
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": source.splitlines(keepends=True),
        }
    )


md("""# Milestone 1: F1 Podium Prediction

## Problem definition
- **Unit of analysis:** one driver in one race (`raceId`, `driverId`).
- **Target:** `podium = 1` if `positionOrder <= 3`.
- **Prediction cutoff:** after qualifying, **before** the race. We never use this race's result fields, pit stops, lap times, or post-race standings as features.

## Evaluation metrics
| Metric | Role |
|--------|------|
| **PR-AUC** | **Primary** — podiums are rare (~15% of rows); PR-AUC reflects ranking quality for the positive class. |
| **ROC-AUC** | Secondary — overall separability. |
| **F1** | Secondary — precision/recall at a threshold chosen on **validation only**. |

Hyperparameters and probability thresholds are tuned on train/validation; **test is untouched** until final reporting.
""")

code(
    """# Kaggle auto-setup (safe locally)
import os
import subprocess
import sys
from pathlib import Path

if Path("/kaggle/input").exists():
    if not os.environ.get("F1_DATA_RAW"):
        for _r in Path("/kaggle/input").rglob("results.csv"):
            _d = _r.parent
            if (_d / "circuits.csv").exists() and (_d / "races.csv").exists():
                os.environ["F1_DATA_RAW"] = str(_d)
                break
    _repo = Path("/kaggle/working/f1-podium-prediction")
    if not (_repo / "src/io.py").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/CrimeWizard/f1-podium-prediction.git", str(_repo)],
            check=True,
        )
    sys.path.insert(0, str(_repo))
    os.chdir(_repo)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn", "shap", "lime"], check=False)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.audit import audit_foreign_keys, audit_primary_keys
from src.cleaning import build_base_results_table
from src.config import DATA_PROCESSED, FIGURES_DIR
from src.de_analysis import (
    question1_front_row_podium_by_circuit,
    question2_home_podium_controlled_grid,
    question3_mechanical_retirement_by_constructor,
)
from src.experiments import (
    ablation_table,
    imputation_strategy_experiment,
    indy_500_exclusion_experiment,
    preprocessing_order_experiments,
)
from src.features import build_modeling_table
from src.inference import predict_podium
from src.io import load_raw_tables, missing_tables, save_processed
from src.splits import temporal_split
from src.status_mapping import build_status_mapping
from src.train import (
    TARGET,
    collect_ffnn_metrics,
    collect_sklearn_metrics,
    feature_columns,
    train_logistic_regression,
    compare_random_forest_overfitting,
    train_random_forest,
    train_shallow_ffnn,
    pick_threshold_on_val,
)
from src.xai import permutation_importance_table

sns.set_theme(style="whitegrid")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
"""
)

md("## 0. Load data")
code(
    """from src.config import get_data_raw

print("Using CSV folder:", get_data_raw())
missing = missing_tables()
if missing:
    raise FileNotFoundError("Missing: " + ", ".join(missing))

tables = load_raw_tables()
list(tables.keys())
"""
)

md(
    """## 1. EDA (before cleaning)

We inspect raw shape, target-related columns, and **qualifying coverage by season** (historical drift motivates median imputation and lagged grid/qualifying features).
"""
)
code(
    """results_raw = tables["results"]
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
"""
)

md(
    """## 2. Auditing keys & cleaning

**Sentinels:** CSV `\\N` parsed as missing via `na_values` in `src/io.py`.

**Grain:** enforce one row per (`raceId`, `driverId`). Early-era **shared drives** create duplicates; we keep the row with the **best `positionOrder`** (lowest value) and document dropped rows in `duplicate_audit`.

**Indianapolis 500 (1950–1960):** excluded from the modeling base (optional sensitivity experiment in §7) because it is not comparable to road-course F1.

**Immutability:** raw files in `data/raw/` are never overwritten; cleaning writes to `data/processed/`.
"""
)
code(
    """pk = pd.DataFrame([r.__dict__ for r in audit_primary_keys(tables)])
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
"""
)

md("## 3. EDA (after cleaning)")
code(
    """fig, ax = plt.subplots(figsize=(6, 4))
base.groupby("year")["podium"].mean().plot(ax=ax)
ax.set_title("Podium rate by season (after cleaning)")
plt.savefig(FIGURES_DIR / "eda_podium_rate_by_year.png", bbox_inches="tight")
plt.show()
"""
)

md(
    """## 4. Data-engineering questions

**Q1** uses **grid positions 1–2** (front row after penalties), not qualifying position alone. Era split at **2014** (hybrid rules).

**Q2** compares home vs away podiums within **grid bins** so we do not confound home advantage with better qualifying.

**Q3** mechanical retirements use an explicit **status → category** map (`src/status_mapping.py`); only `category == mechanical` counts, excluding crashes/driver errors/regulatory DNFs.
"""
)
code(
    """status_map = build_status_mapping(tables["status"])
status_map.to_csv(DATA_PROCESSED / "status_category_mapping.csv", index=False)
display(status_map.groupby("category").size().rename("count"))

q1 = question1_front_row_podium_by_circuit(base, tables["circuits"])
q1_top = q1.sort_values("podium_rate", ascending=False).groupby("era").head(10)
display(q1_top)

plt.figure(figsize=(10, 5))
for era, g in q1.groupby("era"):
    g.nlargest(15, "podium_rate").plot(x="name", y="podium_rate", kind="bar", title=f"Q1 top circuits {era}")
plt.savefig(FIGURES_DIR / "de_q1_front_row_podium.png", bbox_inches="tight")
plt.show()

q2 = question2_home_podium_controlled_grid(base, tables["drivers"], tables["circuits"])
display(q2)
q2.pivot(index="grid_bin", columns="home_race", values="podium_rate").plot(
    kind="bar", figsize=(8, 4), title="Q2 home vs away by grid bin"
)
plt.savefig(FIGURES_DIR / "de_q2_home_advantage.png", bbox_inches="tight")
plt.show()

q3 = question3_mechanical_retirement_by_constructor(base, tables["status"], tables["constructors"])
display(q3.sort_values("mech_rate", ascending=False).groupby("era").head(10))
"""
)

md(
    """## 5. Feature engineering (leakage-safe)

| Feature group | Valid at cutoff because |
|---------------|-------------------------|
| `grid`, qualifying times/position | Set after qualifying, before race |
| `*_before` standings | Shifted to championship state **entering** this race |
| `podium_rate_prev` | Rolling podiums in **prior** races only |

Target `podium` is derived from `positionOrder` but used only as the label, never as an input.
"""
)
code(
    """model_df = build_modeling_table(tables, base)
save_processed(model_df, "modeling_table")
model_df.head()
"""
)

md(
    """## 6. Train / validation / test split

**Temporal split** (no random shuffle — avoids future seasons leaking into training):
- Train: seasons **≤ 2019**
- Validation: **2020–2021** (COVID / format change stress-test)
- Test: **≥ 2022** (ground-effect era, held out until final evaluation)
"""
)
code(
    """train_df, val_df, test_df = temporal_split(model_df)
print("rows:", len(train_df), len(val_df), len(test_df))
print("podium rate %:", train_df[TARGET].mean() * 100, val_df[TARGET].mean() * 100, test_df[TARGET].mean() * 100)
"""
)

md(
    """## 7. Preprocessing experiments (≥2)

1. **Pipeline order:** median impute → scale vs scale → impute (numeric block).
2. **Imputation statistic:** median vs mean.
3. **Indy 500 policy:** exclude vs include championship rounds.

We report validation **PR-AUC** for each variant (primary metric).
"""
)
code(
    """prep_order = preprocessing_order_experiments(train_df, val_df)
prep_impute = imputation_strategy_experiment(train_df, val_df)
prep_indy = indy_500_exclusion_experiment(tables)
display(prep_order)
display(prep_impute)
display(prep_indy)
"""
)

md(
    """## 8. Modeling (≥3 attempts)

1. **Baseline:** logistic regression (interpretable, strong with grid signal).
2. **Non-linear:** random forest (200 trees, class weights).
3. **Shallow FFNN:** 64 → 32 units, dropout 0.2 (Keras).

Threshold for F1: maximized on **validation** F1 per model.
"""
)
code(
    """feats = feature_columns(train_df)
splits = {
    "train": (train_df[feats], train_df[TARGET]),
    "val": (val_df[feats], val_df[TARGET]),
    "test": (test_df[feats], test_df[TARGET]),
}

lr, lr_thr = train_logistic_regression(*splits["train"], *splits["val"])
rf = train_random_forest(*splits["train"])
rf_thr = pick_threshold_on_val(splits["val"][1], rf.predict_proba(splits["val"][0])[:, 1])

metrics = collect_sklearn_metrics("logistic_regression", lr, splits, lr_thr)
metrics += collect_sklearn_metrics("random_forest", rf, splits, rf_thr)

try:
    nn, nn_prep, nn_thr = train_shallow_ffnn(*splits["train"], *splits["val"])
    metrics += collect_ffnn_metrics("shallow_ffnn", nn, nn_prep, splits, nn_thr)
except Exception as e:
    print("FFNN skipped:", e)

metrics_df = pd.DataFrame([m.__dict__ for m in metrics])
metrics_df.to_csv(DATA_PROCESSED / "model_metrics.csv", index=False)
display(metrics_df.pivot_table(index="model", columns="split", values=["roc_auc", "pr_auc", "f1"]))
"""
)

md(
    """## 8b. Random forest — overfitting experiment

Default RF can memorize the training era (very high train PR-AUC). We compare a **regularized** forest:
`max_depth=12`, `min_samples_leaf=25`, `max_samples=0.7`, `max_features="sqrt"`.

**Goal:** lower train–val gap without hurting **validation PR-AUC** (primary metric).
"""
)
code(
    """rf_cmp = compare_random_forest_overfitting(splits)
rf_cmp.to_csv(DATA_PROCESSED / "rf_regularization_comparison.csv", index=False)
display(rf_cmp.pivot_table(index="model", columns="split", values="pr_auc"))
display(rf_cmp.groupby("model")["train_minus_val_pr_auc"].first().rename("train_minus_val_pr_auc"))
"""
)

md("## 9. Feature-group ablation")
code(
    """abl = ablation_table(train_df, val_df, test_df)
abl.to_csv(DATA_PROCESSED / "feature_ablation.csv", index=False)
display(abl)
"""
)

md(
    """## 10. Explainability (XAI)

- **Global:** permutation importance + SHAP beeswarm (validation sample).
- **Local:** LIME for one validation row.

**Causal disclaimer:** these tools describe what the model relied on for a prediction, not what *causes* a podium in the real world.
"""
)
code(
    """imp = permutation_importance_table(lr, splits["val"][0], splits["val"][1])
display(imp.head(15))

try:
    from src.xai import shap_summary
    import shap

    shap_values = shap_summary(lr, splits["val"][0].sample(min(500, len(val_df)), random_state=42))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.savefig(FIGURES_DIR / "xai_shap_beeswarm.png", bbox_inches="tight")
    plt.show()
except Exception as e:
    print("SHAP skipped:", e)

try:
    from src.xai import lime_explain_row

    row = splits["val"][0].iloc[[0]]
    exp = lime_explain_row(lr, row, splits["train"][0].sample(min(2000, len(train_df)), random_state=0))
    exp.as_pyplot_figure()
    plt.savefig(FIGURES_DIR / "xai_lime_row0.png", bbox_inches="tight")
    plt.show()
except Exception as e:
    print("LIME skipped:", e)
"""
)

md("## 11. Inference function demo")
code(
    """complete = splits["val"][0].iloc[0].to_dict()
partial = {feats[0]: complete[feats[0]]}
print("Complete row:", predict_podium(lr, complete, threshold=lr_thr, feature_names=feats))
print("Partial row:", predict_podium(lr, partial, threshold=lr_thr, feature_names=feats))
"""
)

md(
    """## 12. Known limitations

- Qualifying features are sparse before the mid-1990s; grid and standings partially substitute.
- Constructor renames break historical "team form" continuity (`constructorId` resets).
- Sprint weekends (2021+) introduce judgment calls; we did not add sprint results as features in this baseline.
- Class imbalance persists; PR-AUC and calibrated thresholds matter more than raw accuracy.
"""
)

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
