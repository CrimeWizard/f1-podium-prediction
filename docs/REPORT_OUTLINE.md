# Analytical Report Outline (export to PDF)

1. **Introduction** — task, cutoff, dataset grain `(raceId, driverId)`.
2. **EDA & cleaning** — before/after plots; `\N` handling; duplicate policy; FK audit summary.
3. **Data-engineering answers**
   - Q1: Front-row → podium by circuit; pre/post 2014; grid vs qualifying.
   - Q2: Home race effect controlling for grid bins.
   - Q3: Mechanical retirement rates by constructor (2014–2021 vs 2022–2024); status mapping table.
4. **Features** — list each feature and why it is valid pre-race; qualifying coverage limitation.
5. **Preprocessing experiments** — at least two order/choice comparisons with metrics.
6. **Modeling** — baseline + ≥2 statistical models + shallow NN; metric table (ROC-AUC, PR-AUC, F1) on val & test.
7. **Ablation** — feature groups removed; interpretation.
8. **XAI** — permutation importance, SHAP global, LIME/local SHAP; non-causal disclaimer.
9. **Limitations** — era drift, constructor renames, sprint weekends, missing qualifying history.
10. **Inference** — example full row + partial row.
