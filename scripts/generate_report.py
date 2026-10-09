#!/usr/bin/env python3
"""Build docs/ANALYTICAL_REPORT.md from latest pipeline outputs."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
metrics_path = ROOT / "data" / "processed" / "model_metrics.csv"
abl_path = ROOT / "data" / "processed" / "feature_ablation.csv"
status_path = ROOT / "data" / "processed" / "status_category_mapping.csv"
out = ROOT / "docs" / "ANALYTICAL_REPORT.md"


def main() -> None:
    if not metrics_path.exists():
        raise SystemExit("Run notebooks/milestone1.ipynb first (missing model_metrics.csv).")

    metrics = pd.read_csv(metrics_path)
    abl = pd.read_csv(abl_path) if abl_path.exists() else None
    status_counts = (
        pd.read_csv(status_path).groupby("category").size().to_string()
        if status_path.exists()
        else "(run notebook §4)"
    )

    pivot = metrics.pivot_table(index="model", columns="split", values=["pr_auc", "roc_auc", "f1"])

    body = f"""# Analytical Report — F1 Podium Prediction (Milestone 1)

> Auto-generated skeleton from `data/processed/*.csv`. Export to PDF for submission and expand prose with team interpretation.

## 1. Task & cutoff
We predict whether each driver finishes on the podium (`positionOrder ≤ 3`) using only information available **after qualifying and before the race**. Post-race fields (points, status, pit stops, lap times, updated standings) are labels or analysis-only, never features.

**Primary metric:** PR-AUC (rare positive class). **Secondary:** ROC-AUC and F1 at a validation-tuned threshold.

## 2. Cleaning summary
- Sentinels `\\\\N` → missing; lap times converted to seconds in qualifying.
- Grain: one row per (`raceId`, `driverId`); shared-drive duplicates resolved by **best `positionOrder`**.
- Indianapolis 500 rounds (1950–1960) excluded from the main modeling table (see preprocessing experiment for sensitivity).
- Primary/foreign key audits in notebook §2.

## 3. Data-engineering answers
### Q1 — Front row → podium by circuit (pre/post 2014)
See `outputs/figures/de_q1_front_row_podium.png`. Front row defined by **grid** positions 1–2.

### Q2 — Home race advantage controlling for grid
See `outputs/figures/de_q2_home_advantage.png`. Home = driver nationality matches circuit country (with normalization); compared within grid bins.

### Q3 — Mechanical retirement rate by constructor
Status categories (counts):

```
{status_counts}
```

Mechanical rate = mechanical DNFs / race starts (grid > 0), eras 2014–2021 vs 2022–2024. Figure/table in notebook §4.

## 4. Features (leakage-safe)
Grid & qualifying, championship standings shifted to **before race**, rolling driver podium rate from **prior** races only. See notebook §5 table.

## 5. Model comparison

```
{pivot.to_string()}
```

**Modeling attempts:** (1) logistic regression baseline, (2) random forest, (3) shallow FFNN.

## 6. Feature ablation

```
{abl.to_string() if abl is not None else '(run notebook §9)'}
```

## 7. XAI
Permutation importance, SHAP beeswarm, LIME local plot in `outputs/figures/`. **Not causal:** importance ≠ proof of causal effect on podiums.

## 8. Limitations
Qualifying coverage drift, constructor identity changes, sprint-format heterogeneity, class imbalance.

## 9. Inference
`src/inference.py` — demonstrated on complete and partial rows in notebook §11.
"""
    out.write_text(body)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
