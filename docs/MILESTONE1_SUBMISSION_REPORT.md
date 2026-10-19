# Milestone 1 Submission Report - F1 Podium Prediction

**Run provenance:** Kaggle notebook `ACL_MS1`, Version 7 of 7; successful 113.1-second run with 110 output files. Source revision `3f5da97`; Formula 1 World Championship dataset (1950-2024). The PDF figures and metrics were extracted from the saved Version 7 Kaggle notebook.

**Course:** Introduction to Artificial Intelligence - Milestone 1  
**Team representative:** Youssef Ahmed  
**Team:** Youssef Ahmed (58-4250, T-27); Omar Eldafrawy (58-5371, T-11); Ismail Almoatazbellah (58-4720, T-16); Mohamed Elayadi (43-2918, T-22)  
**Notebook:** https://www.kaggle.com/code/crimewizard/acl-ms1  
**GitHub:** https://github.com/CrimeWizard/f1-podium-prediction  
**Dataset:** Formula 1 World Championship, Ergast-derived tables, 1950-2024.

## Summary

The task predicts whether a driver finishes on the podium from information available after qualifying and before the race. The temporal split is train through 2019 (24,148 rows), validation 2020-2021 (780), and test 2022-2024 (1,359). The regularized random forest achieved the highest test PR-AUC (0.696); the shallow FFNN achieved the highest validation PR-AUC (0.743).

## Data and engineering

Fourteen relational CSV tables were audited. Primary-key audits found zero duplicate or null keys; checked foreign-key relationships found zero orphan or null references. Duplicate driver-race entries were resolved by best (lowest) `positionOrder`: 176 duplicate rows in the full analysis base and 131 in the modeling base. Indianapolis 500 rounds are retained for Q1-Q3 and excluded only from the main modeling table; sensitivity results are reported.

Q1 compares qualifying front row and actual-grid front row separately, excludes 1,601 explicit non-starters, and requires at least 20 starts per circuit/era. Highest observed conversion: Magny-Cours pre-2014 actual grid 29/36 (80.6%); Yas Marina from 2014 actual grid 19/22 (86.4%); Suzuka pre-2014 qualifying 20/26 (76.9%); Monaco from 2014 qualifying 17/20 (85.0%). Paired rank correlations over seven shared circuits: -0.180 actual grid, -0.291 qualifying.

Q2 defines home as normalized driver nationality matching circuit country. Rates by grid bin (home vs away): 1-5, 45.8% (430) vs 45.2% (5,180); 6-10, 9.6% (479) vs 11.4% (5,126); 11-15, 3.4% (469) vs 3.0% (5,117); 16+, 0.8% (933) vs 0.7% (7,203). Direction varies, so there is no consistent advantage after grid control. Zero/missing-grid exclusions: 130; unknown-country exclusions: 0.

Q3's highest mechanical-retirement rates: Caterham 2014-2021, 11/34 (32.4%); RB F1 Team 2022-2024, 5/48 (10.4%). Rates are descriptive and denominators modest.

## Modeling and results

The target is `positionOrder <= 3`. Features include grid, qualifying, standings shifted to before the race, and prior-race podium form. Post-race fields are excluded. PR-AUC is primary; thresholds are selected using validation data. The notebook reports ROC-AUC, PR-AUC, F1, accuracy, precision, and recall for train, validation, and test; the last four are computed at the validation-selected threshold.

| Model | Split | PR-AUC | ROC-AUC | F1 | Accuracy | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | Train | 0.536977 | 0.891370 | 0.559466 | 0.879742 | 0.518852 | 0.606978 |
| Logistic regression | Validation | 0.700900 | 0.918114 | 0.687747 | 0.898718 | 0.639706 | 0.743590 |
| Logistic regression | Test | 0.661004 | 0.924586 | 0.666667 | 0.883738 | 0.585185 | 0.774510 |
| Random forest (default) | Train | 0.975409 | 0.997616 | 0.953463 | 0.987784 | 0.915480 | 0.994733 |
| Random forest (default) | Validation | 0.685899 | 0.912751 | 0.689956 | 0.908974 | 0.705357 | 0.675214 |
| Random forest (default) | Test | 0.648018 | 0.925040 | 0.640964 | 0.890361 | 0.630332 | 0.651961 |
| Random forest (regularized) | Train | 0.627737 | 0.924204 | 0.594472 | 0.896720 | 0.587404 | 0.601712 |
| Random forest (regularized) | Validation | 0.707244 | 0.923670 | 0.704348 | 0.912821 | 0.716814 | 0.692308 |
| Random forest (regularized) | Test | 0.696453 | 0.933499 | 0.641975 | 0.893304 | 0.646766 | 0.637255 |
| Shallow FFNN | Train | 0.579885 | 0.904390 | 0.572466 | 0.887858 | 0.550061 | 0.596774 |
| Shallow FFNN | Validation | 0.743255 | 0.909257 | 0.715517 | 0.915385 | 0.721739 | 0.709402 |
| Shallow FFNN | Test | 0.662440 | 0.926360 | 0.668182 | 0.892568 | 0.622881 | 0.720588 |

The PDF includes the notebook's PR-AUC-by-model-and-split chart, in addition to EDA, engineering-question, feature-ablation, permutation-importance, SHAP, and LIME plots. Preprocessing comparisons include impute-then-scale vs scale-then-impute, median vs mean imputation, and inclusion/exclusion of Indianapolis 500. Feature ablation removes grid, qualifying, standings, and form groups. The inference demo covers complete and partial input rows.

## Limitations

Qualifying coverage is sparse in early seasons; constructor identities change; sprint-specific features are not included; mechanical status is inferred from labels without an independent starter flag. Results and explainability are descriptive, not causal. Circuit and constructor leaders have modest denominators.
