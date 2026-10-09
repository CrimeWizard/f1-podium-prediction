# Analytical Report — F1 Podium Prediction (Milestone 1)

> Auto-generated skeleton from `data/processed/*.csv`. Export to PDF for submission and expand prose with team interpretation.

## 1. Task & cutoff
We predict whether each driver finishes on the podium (`positionOrder ≤ 3`) using only information available **after qualifying and before the race**. Post-race fields (points, status, pit stops, lap times, updated standings) are labels or analysis-only, never features.

**Primary metric:** PR-AUC (rare positive class). **Secondary:** ROC-AUC and F1 at a validation-tuned threshold.

## 2. Cleaning summary
- Sentinels `\\N` → missing; lap times converted to seconds in qualifying.
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
category
accident_or_driver    16
finished              34
mechanical            45
other_dnf             37
regulatory             7
```

Mechanical rate = mechanical DNFs / race starts (grid > 0), eras 2014–2021 vs 2022–2024. Figure/table in notebook §4.

## 4. Features (leakage-safe)
Grid & qualifying, championship standings shifted to **before race**, rolling driver podium rate from **prior** races only. See notebook §5 table.

## 5. Model comparison

```
                           f1                        pr_auc                       roc_auc                    
split                    test     train       val      test     train       val      test     train       val
model                                                                                                        
logistic_regression  0.640351  0.555131  0.691358  0.669177  0.545584  0.706703  0.925677  0.893458  0.919674
random_forest        0.629723  0.987803  0.690909  0.667523  0.998271  0.708889  0.926146  0.999799  0.920105
shallow_ffnn         0.651376  0.569825  0.716157  0.666446  0.583159  0.735924  0.923186  0.905844  0.913125
```

**Modeling attempts:** (1) logistic regression baseline, (2) random forest, (3) shallow FFNN.

## 6. Feature ablation

```
  feature_group_removed  pr_auc_val  pr_auc_test
0                  none    0.706703     0.669177
1                  grid    0.683260     0.643773
2            qualifying    0.701430     0.650583
3             standings    0.701183     0.656472
4                  form    0.711875     0.650220
```

## 7. XAI
Permutation importance, SHAP beeswarm, LIME local plot in `outputs/figures/`. **Not causal:** importance ≠ proof of causal effect on podiums.

## 8. Limitations
Qualifying coverage drift, constructor identity changes, sprint-format heterogeneity, class imbalance.

## 9. Inference
`src/inference.py` — demonstrated on complete and partial rows in notebook §11.
