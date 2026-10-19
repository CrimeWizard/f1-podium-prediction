# Milestone 1 Submission Report - F1 Podium Prediction

**Run provenance:** Kaggle notebook `ACL_MS1`, Version 6 of 6; successful 99.7-second run with 110 output files. Source revision `7a540b3`; Formula 1 World Championship dataset (1950-2024). Live metrics below were checked against the Kaggle Version 6 page. The PDF embeds notebook-rendered plots.

**Course:** Introduction to Artificial Intelligence - Milestone 1  
**Team representative:** Youssef Ahmed  
**Team:** Youssef Ahmed (58-4250, T-27); Omar Eldafrawy (58-5371, T-11); Ismail Almoatazbellah (58-4720, T-16); Mohamed Elayadi (43-2918, T-22)  
**Notebook:** https://www.kaggle.com/code/crimewizard/acl-ms1  
**GitHub:** https://github.com/CrimeWizard/f1-podium-prediction  
**Dataset:** Formula 1 World Championship, Ergast-derived tables, 1950-2024.

## Summary

The task predicts whether a driver finishes on the podium from information available after qualifying and before the race. The temporal split is train through 2019 (24,148 rows), validation 2020-2021 (780), and test 2022-2024 (1,359). The regularized random forest achieved the highest held-out test PR-AUC (0.696); shallow FFNN had highest validation PR-AUC (0.736).

## Data and engineering

Fourteen relational CSV tables were audited. Primary-key audits found zero duplicate or null keys; checked foreign-key relationships found zero orphan or null references. Duplicate driver-race entries were resolved by best (lowest) `positionOrder`: 176 duplicate rows in the full analysis base and 131 in the modeling base. Indianapolis 500 rounds are retained for Q1-Q3 and excluded only from the main modeling table; sensitivity results are reported.

Q1 compares qualifying front row and actual-grid front row separately, excludes 1,601 explicit non-starters, and requires at least 20 starts per circuit/era. Highest observed conversion: Magny-Cours pre-2014 actual grid 29/36 (80.6%); Yas Marina from 2014 actual grid 19/22 (86.4%); Suzuka pre-2014 qualifying 20/26 (76.9%); Monaco from 2014 qualifying 17/20 (85.0%). Paired rank correlations over seven shared circuits: -0.180 actual grid, -0.291 qualifying.

Q2 defines home as normalized driver nationality matching circuit country. Rates by grid bin (home vs away): 1-5, 45.8% (430) vs 45.2% (5,180); 6-10, 9.6% (479) vs 11.4% (5,126); 11-15, 3.4% (469) vs 3.0% (5,117); 16+, 0.8% (933) vs 0.7% (7,203). Direction varies, so there is no consistent advantage after grid control. Zero/missing-grid exclusions: 130; unknown-country exclusions: 0.

Q3's highest mechanical-retirement rates: Caterham 2014-2021, 11/34 (32.4%); RB F1 Team 2022-2024, 5/48 (10.4%). Rates are descriptive and denominators modest.

## Modeling and results

The target is `positionOrder <= 3`. Features include grid, qualifying, standings shifted to before the race, and prior-race podium form. Post-race fields are excluded. PR-AUC is primary; F1 thresholds are selected on validation data.

| Model | Split | PR-AUC | ROC-AUC | F1 |
|---|---|---:|---:|---:|
| Logistic regression | Train | 0.536977 | 0.891370 | 0.559466 |
| Logistic regression | Validation | 0.700900 | 0.918114 | 0.687747 |
| Logistic regression | Test | 0.661004 | 0.924586 | 0.666667 |
| Random forest (default) | Train | 0.975409 | 0.997616 | 0.953463 |
| Random forest (default) | Validation | 0.685899 | 0.912751 | 0.689956 |
| Random forest (default) | Test | 0.648018 | 0.925040 | 0.640964 |
| Random forest (regularized) | Train | 0.627737 | 0.924204 | 0.594472 |
| Random forest (regularized) | Validation | 0.707244 | 0.923670 | 0.704348 |
| Random forest (regularized) | Test | 0.696453 | 0.933499 | 0.641975 |
| Shallow FFNN | Train | 0.576309 | 0.903944 | 0.554541 |
| Shallow FFNN | Validation | 0.735562 | 0.908213 | 0.719298 |
| Shallow FFNN | Test | 0.667145 | 0.924357 | 0.643192 |

Preprocessing comparisons include impute-then-scale vs scale-then-impute, median vs mean imputation, and inclusion/exclusion of Indianapolis 500. Feature ablation removes grid, qualifying, standings, and form groups in turn. Explainability includes permutation importance, SHAP and LIME. The inference demo covers complete and partial input rows.

## Limitations

Qualifying coverage is sparse in early seasons; constructor identities change; sprint-specific features are not included; mechanical status is inferred from labels without an independent starter flag. Results and explainability are descriptive, not causal. Circuit and constructor leaders have modest denominators.
