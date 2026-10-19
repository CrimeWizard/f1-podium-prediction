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
from IPython.display import Markdown, display

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
expected_tables = {Path(p).stem for p in __import__("src.config", fromlist=["TABLE_FILES"]).TABLE_FILES}
assert set(tables) == expected_tables, f"Unexpected/missing raw tables: {expected_tables.symmetric_difference(tables)}"
assert set(pk["table"]) == expected_tables, f"Primary-key audit did not cover: {expected_tables.difference(pk['table'])}"
assert not pk["duplicate_rows"].any() and not pk["null_key_rows"].any(), "Primary-key audit failed"
assert not fk["orphan_rows"].any(), "Foreign-key audit failed"
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

**Q1** reports qualifying positions 1–2 and actual grid positions 1–2 separately: penalties can make them differ. Both denominators include actual starters only; missing qualifying records are excluded from the qualifying comparison. The ranking chart requires at least 20 starts per circuit/era/basis to reduce small-sample noise; full counts remain in the saved table.

**Q2** compares home vs away podium rates within valid actual-grid bins (1–5, 6–10, 11–15, 16+). Missing/zero grids and rows with unknown driver nationality or circuit country are excluded.

**Q3** mechanical retirements use an explicit **status → category** map (`src/status_mapping.py`); only `category == mechanical` counts. Explicit pre-race non-start statuses are excluded from the denominator; grid 0 is retained because it can mean a pit-lane start. Ergast has no independent starter flag, so a mechanical failure recorded without an explicit non-start status can be ambiguous at the start line. Sample counts are shown because constructor rates can be noisy for small teams.
"""
)
code(
    """status_map = build_status_mapping(tables["status"])
status_map.to_csv(DATA_PROCESSED / "status_category_mapping.csv", index=False)
display(status_map.groupby("category").size().rename("count"))
display(status_map)

q1 = question1_front_row_podium_by_circuit(base, tables["circuits"], tables["qualifying"], tables["status"])
q1.to_csv(DATA_PROCESSED / "de_q1_front_row_by_circuit.csv", index=False)
print("Q1 nonstarters excluded:", q1.attrs["nonstarters_excluded"])
print("Q1 starters without a valid qualifying position:", q1.attrs["starters_without_qualifying_position"])
q1_stable = q1.loc[q1["front_row_starts"] >= 20].copy()
q1_stable["rank_within_era"] = q1_stable.groupby(["start_basis", "era"])["podium_rate"].rank(method="min", ascending=False)
q1_rank = q1_stable.pivot_table(index=["start_basis", "circuitId", "name"], columns="era", values="rank_within_era")
if "pre_2014" in q1_rank and "from_2014" in q1_rank:
    q1_rank["rank_change_pre_minus_post"] = q1_rank["pre_2014"] - q1_rank["from_2014"]
q1_rank.to_csv(DATA_PROCESSED / "de_q1_circuit_rank_changes.csv")
q1_top = q1_stable.sort_values("podium_rate", ascending=False).groupby(["start_basis", "era"], observed=True).head(10)
display(q1_top.sort_values(["start_basis", "era", "podium_rate"], ascending=[True, True, False]))
q1_lines = []
for (basis, era), group in q1_top.groupby(["start_basis", "era"], observed=True):
    best = group.sort_values("podium_rate", ascending=False).iloc[0]
    q1_lines.append(
        f"- **{basis.replace('_', ' ')} / {era}:** {best['name']} "
        f"({best['podium_rate']:.1%}; {int(best['podiums'])}/{int(best['front_row_starts'])} starters)."
    )
display(Markdown("**Answer Q1.** Highest observed conversion among circuits with at least 20 relevant starts:\\n\\n" + "\\n".join(q1_lines)))
for basis, ranks in q1_rank.groupby(level="start_basis"):
    paired = ranks.dropna(subset=["pre_2014", "from_2014"])
    if len(paired) >= 2:
        rho = paired["pre_2014"].corr(paired["from_2014"], method="spearman")
        print(f"{basis}: paired-circuit Spearman rank correlation pre/from 2014 = {rho:.3f} (n={len(paired)} circuits)")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, ((basis, era), g) in zip(axes.flat, q1_stable.groupby(["start_basis", "era"], observed=True)):
    top = g.nlargest(10, "podium_rate").sort_values("podium_rate")
    ax.barh(top["name"], top["podium_rate"])
    ax.set_title(f"{basis.replace('_', ' ').title()} — {era}")
    ax.set_xlabel("Podium rate")
    ax.set_xlim(0, 1)
fig.suptitle("Q1 Front-row conversion by circuit: qualifying vs actual grid")
fig.tight_layout()
plt.savefig(FIGURES_DIR / "de_q1_front_row_podium.png", bbox_inches="tight")
plt.show()

q2 = question2_home_podium_controlled_grid(base, tables["drivers"], tables["circuits"], tables["status"])
print("Q2 rows excluded for unknown country:", q2.attrs["unknown_country_rows_excluded"])
print("Q2 rows excluded for zero/missing grid:", q2.attrs["zero_or_missing_grid_rows_excluded"])
display(q2)
q2_pivot = q2.pivot(index="grid_bin", columns="home_race", values="podium_rate")
q2_counts = q2.pivot(index="grid_bin", columns="home_race", values="starts")
q2_lines = []
q2_differences = []
for grid_bin in q2_pivot.index:
    away = q2_pivot.loc[grid_bin].get(0, float("nan"))
    home = q2_pivot.loc[grid_bin].get(1, float("nan"))
    n_away = q2_counts.loc[grid_bin].get(0, 0)
    n_home = q2_counts.loc[grid_bin].get(1, 0)
    if pd.notna(home) and pd.notna(away):
        q2_differences.append(home - away)
        q2_lines.append(f"- **Grid {grid_bin}:** home {home:.1%} (n={int(n_home)}) vs away {away:.1%} (n={int(n_away)}).")
if q2_differences and all(d > 0 for d in q2_differences):
    q2_summary = "Home drivers have a higher observed podium rate in every compared grid bin."
elif q2_differences and all(d < 0 for d in q2_differences):
    q2_summary = "Home drivers have a lower observed podium rate in every compared grid bin."
else:
    q2_summary = "The direction varies by grid bin; the data do not show a consistent home advantage after this control."
display(Markdown("**Answer Q2.** " + q2_summary + "\\n\\n" + "\\n".join(q2_lines)))
q2_pivot.plot(
    kind="bar", figsize=(8, 4), title="Q2 home vs away by grid bin"
)
plt.savefig(FIGURES_DIR / "de_q2_home_advantage.png", bbox_inches="tight")
plt.show()

q3 = question3_mechanical_retirement_by_constructor(base, tables["status"], tables["constructors"])
q3.to_csv(DATA_PROCESSED / "de_q3_mechanical_retirement_by_constructor.csv", index=False)
print("Q3 explicit nonstarters excluded:", q3.attrs["nonstarters_excluded"])
q3_top = q3.sort_values("mech_rate", ascending=False).groupby("era", observed=True).head(10)
display(q3_top.sort_values(["era", "mech_rate"], ascending=[True, False]))
q3_lines = []
for era, group in q3_top.groupby("era", observed=True):
    best = group.sort_values("mech_rate", ascending=False).iloc[0]
    q3_lines.append(
        f"- **{era}:** {best['name']} has the highest observed rate "
        f"({best['mech_rate']:.1%}; {int(best['mechanical_retirements'])}/"
        f"{int(best['starts'])} starters)."
    )
display(Markdown("**Answer Q3.** Highest observed mechanical-retirement rate by era (rates are descriptive; denominators differ):\\n\\n" + "\\n".join(q3_lines)))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, (era, g) in zip(axes, q3.groupby("era", observed=True)):
    top = g.nlargest(10, "mech_rate").sort_values("mech_rate")
    ax.barh(top["name"], top["mech_rate"])
    ax.set_title(f"{era}")
    ax.set_xlabel("Mechanical retirements / starts")
    ax.set_xlim(0, max(0.05, float(q3_top["mech_rate"].max()) * 1.1))
fig.suptitle("Q3 Constructors with highest mechanical-retirement rates")
fig.tight_layout()
plt.savefig(FIGURES_DIR / "de_q3_mechanical_retirement.png", bbox_inches="tight")
plt.show()
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

md("""### Feature evidence before modeling

Grid is measured after penalties and is available at the prediction cutoff. Prior podium form is shifted so it excludes the current race. These plots check their relationship to the target before fitting models; they are descriptive, not causal.
""")
code(
    """grid_eda = model_df.loc[model_df["grid"].between(1, 30)].copy()
grid_eda = grid_eda.groupby("grid", observed=True).agg(starts=(TARGET, "size"), podium_rate=(TARGET, "mean"))
grid_eda = grid_eda.loc[grid_eda["starts"] >= 20]
form_eda = model_df.dropna(subset=["podium_rate_prev"]).copy()
form_eda["prior_form_bin"] = pd.qcut(form_eda["podium_rate_prev"], q=5, duplicates="drop")
form_eda = form_eda.groupby("prior_form_bin", observed=True).agg(starts=(TARGET, "size"), podium_rate=(TARGET, "mean"))
display(grid_eda)
display(form_eda)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
grid_eda["podium_rate"].plot(marker="o", ax=axes[0], title="Podium rate by actual grid position")
axes[0].set_ylabel("Podium rate")
form_eda["podium_rate"].plot(kind="bar", ax=axes[1], title="Podium rate by prior-form quantile group")
axes[1].set_ylabel("Podium rate")
fig.tight_layout()
plt.savefig(FIGURES_DIR / "feature_target_relationships.png", bbox_inches="tight")
plt.show()
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

We report validation **PR-AUC** for each variant (primary metric); result tables are saved under `data/processed/`.
"""
)
code(
    """prep_order = preprocessing_order_experiments(train_df, val_df)
prep_impute = imputation_strategy_experiment(train_df, val_df)
prep_indy = indy_500_exclusion_experiment(tables)
display(prep_order)
display(prep_impute)
display(prep_indy)
pd.concat([
    prep_order.rename(columns={"preprocessing_order": "experiment"}).assign(experiment=lambda d: "order: " + d["experiment"]),
    prep_impute.rename(columns={"imputation_strategy": "experiment"}).assign(experiment=lambda d: "imputation: " + d["experiment"]),
    prep_indy.rename(columns={"indy_policy": "experiment"}).assign(experiment=lambda d: "Indianapolis: " + d["experiment"]),
], ignore_index=True)[["experiment", "pr_auc_val"]].to_csv(DATA_PROCESSED / "preprocessing_experiments.csv", index=False)

# Show the numerical effect of median imputation followed by standardization.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
preview_cols = [c for c in ["grid", "q1_sec", "points_before", "podium_rate_prev"] if c in train_df]
preview = train_df[preview_cols]
imputed_preview = SimpleImputer(strategy="median").fit_transform(preview)
scaled_preview = StandardScaler().fit_transform(imputed_preview)
distribution_effect = pd.DataFrame({
    "feature": preview_cols,
    "missing_pct_before": [preview[c].isna().mean() * 100 for c in preview_cols],
    "raw_min": [preview[c].min() for c in preview_cols],
    "raw_max": [preview[c].max() for c in preview_cols],
    "scaled_mean_after": scaled_preview.mean(axis=0),
    "scaled_std_after": scaled_preview.std(axis=0),
})
display(distribution_effect)
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
rf_regularized_metrics = rf_cmp.loc[
    rf_cmp["model"] == "random_forest_regularized",
    ["model", "split", "roc_auc", "pr_auc", "f1"],
]
metrics_df = pd.concat([metrics_df, rf_regularized_metrics], ignore_index=True)
metrics_df.to_csv(DATA_PROCESSED / "model_metrics.csv", index=False)
display(metrics_df.pivot_table(index="model", columns="split", values=["roc_auc", "pr_auc", "f1"]))
"""
)

md("""## 9. Feature-group ablation

Each feature group is removed in turn and the same logistic-regression pipeline is retrained. Compare validation and held-out test PR-AUC with the full-feature baseline; test values are reported only after the final evaluation.
""")
code(
    """abl = ablation_table(train_df, val_df, test_df)
abl.to_csv(DATA_PROCESSED / "feature_ablation.csv", index=False)
display(abl)
abl[abl["feature_group_removed"] != "none"].sort_values("pr_auc_test").plot(
    x="feature_group_removed", y="pr_auc_test", kind="bar", legend=False,
    title="Test PR-AUC after removing each feature group", ylabel="PR-AUC"
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_ablation_test_pr_auc.png", bbox_inches="tight")
plt.show()
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
imp.to_csv(DATA_PROCESSED / "permutation_importance.csv", index=False)
display(imp.head(15))
imp.head(15).sort_values("importance_mean").plot.barh(
    x="feature", y="importance_mean", xerr="importance_std", legend=False,
    title="Permutation importance on validation data (PR-AUC scoring)"
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "xai_permutation_importance.png", bbox_inches="tight")
plt.show()

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
