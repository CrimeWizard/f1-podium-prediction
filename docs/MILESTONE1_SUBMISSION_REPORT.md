# Milestone 1 Submission Report - F1 Podium Prediction

**Run Provenance:** Kaggle notebook `ACL_MS1`, Version 7 of 7; successful 113.1-second run with 110 output files. Source revision `3f5da97`; Formula 1 World Championship dataset (1950–2024). All PDF figures and metrics were extracted directly from the saved Version 7 Kaggle notebook.

**Course:** Introduction to Artificial Intelligence — Milestone 1  
**Team Representative:** Youssef Ahmed  
**Team Members:** Youssef Ahmed (58-4250, T-27); Omar Eldafrawy (58-5371, T-11); Ismail Almoatazbellah (58-4720, T-16); Mohamed Elayadi (43-2918, T-22)  
**Kaggle Notebook:** https://www.kaggle.com/code/crimewizard/acl-ms1  
**GitHub Repository:** https://github.com/CrimeWizard/f1-podium-prediction  
**Dataset:** Formula 1 World Championship, Ergast-derived relational tables (1950–2024).

---

## Executive Summary

This project builds an end-to-end, leakage-safe machine learning pipeline to predict whether a Formula 1 driver will finish on the podium (`positionOrder <= 3`) based strictly on information available after Saturday qualifying and before Sunday race start. Using 14 relational tables from the Kaggle Ergast dataset (1950–2024), we conduct extensive data auditing, address historical shared-drive duplicates, and engineer pre-race championship standings and rolling form features. A strictly chronological train/validation/test split is enforced (train &le; 2019, validation 2020–2021, held-out test 2022–2024). Given severe class imbalance (~13–15% podiums), PR-AUC is our primary evaluation metric. Among four model variants, regularized Random Forest attained the highest unseen test PR-AUC (0.6965), while a shallow Neural Network achieved the highest validation PR-AUC (0.7433). Both decisively outperformed a defaulted, overfitted tree baseline and linear logistic regression.

---

## 1. Task, Unit of Analysis, and Prediction Cutoff

- **Unit of Analysis:** One driver-race entry uniquely identified by `(raceId, driverId)`.
- **Target:** Binary indicator `positionOrder <= 3` (podium finish).
- **Prediction Cutoff:** Strictly after Saturday qualifying and before Sunday race start.
- **Leakage Prevention:** All post-cutoff attributes (race finish position, points scored, laps completed, pit stops, DNF status, post-race standings) are excluded from feature sets.

---

## 2. Data Auditing, Cleaning, and Duplicate Handling

- **Ingestion & Sentinels:** 14 relational CSV tables loaded immutably. Sentinels (literal `\N`) converted to nulls; dates parsed to datetimes; qualifying times parsed to numeric seconds.
- **Key Audits:** Primary keys verified 100% unique and non-null across all 14 tables (0 violations). Checked 23 foreign-key relationships; 0 orphan rows and 0 broken links.
- **Shared-Drive Reconciliation:** 1950s car swaps mid-race produced duplicate `(raceId, driverId)` rows. Found 176 duplicate rows in the full dataset (131 when excluding the non-championship Indianapolis 500 rounds). Kept the row with the best (lowest) `positionOrder`.
- **Final Modeling Table:** 26,287 unique driver-race observations.

---

## 3. Data-Engineering Questions (Multi-Hop Joins)

### Q1. Front-Row Starts and Podium Conversion (Pre- vs Post-2014)
- Distinguishes starting grid front rows (grid 1–2, post-penalties) from qualifying session front rows (quali 1–2).
- Multi-hop join: `results` + `races` + `circuits` + `qualifying`. Excludes 1,601 non-starters and accounts for 14,231 legacy starters lacking qualifying records. Minimum &ge; 20 starts filter applied.
- **Top Conversions:**
  - *Actual Grid Front Row (Pre-2014):* Magny-Cours converts highest at 80.6% (29/36).
  - *Actual Grid Front Row (From 2014):* Yas Marina converts highest at 86.4% (19/22).
  - *Qualifying Front Row (Pre-2014):* Suzuka converts highest at 76.9% (20/26).
  - *Qualifying Front Row (From 2014):* Monaco converts highest at 85.0% (17/20).
- **Era Correlation:** Paired Spearman rank correlation across 7 common circuits is weakly negative: -0.180 (actual grid) and -0.291 (qualifying), demonstrating hybrid-era aerodynamic and DRS regulation shifts.

### Q2. Home-Race Performance Controlling for Starting Grid Position
- Driver nationality normalized and matched against circuit host country. Stratified by starting grid bins (1–5, 6–10, 11–15, 16+) to decouple home advantage from qualifying speed.
- Exclusions: 0 unknown country rows, 130 zero/missing grid rows.
- **Rates by Grid Bin:**
  - *Grid 1–5:* Home 48.7% (181/372) vs Away 45.2% (2,340/5,180) (+3.5%)
  - *Grid 6–10:* Home 8.7% (36/416) vs Away 11.4% (582/5,126) (-2.7%)
  - *Grid 11–15:* Home 2.7% (11/414) vs Away 3.0% (154/5,117) (-0.3%)
  - *Grid 16+:* Home 0.3% (2/729) vs Away 0.7% (53/7,202) (-0.4%)
- **Conclusion:** Home advantage is inconsistent across bins and negligible in the midfield/rear, indicating that raw apparent home advantage is driven by qualifying grid slot rather than home-track psychology.

### Q3. Mechanical Retirement Rates by Constructor (2014–2021 vs 2022–2024)
- Classified status strings into pure mechanical failures (power unit, gearbox, hydraulics, electrical, suspension), excluding accidents, driver errors, and regulatory disqualifications.
- **Findings:**
  - *2014–2021 (Turbo-Hybrid):* Caterham highest at 32.4% (11/34 starts).
  - *2022–2024 (Ground-Effect):* RB F1 Team highest at 10.4% (5/48 starts).
- Mechanical retirements dropped significantly across all constructors in 2022–2024 due to mature engine freeze rules.

---

## 4. Leakage-Safe Feature Selection

| Feature Group | Description and Pre-Race Validity |
|---|---|
| `grid` | Official starting grid slot after steward penalties; strongly predictive. |
| `quali_position`, `q1_sec`, `q2_sec`, `q3_sec` | Qualifying session ranks and lap times in seconds (available post-1994). |
| `points_before`, `position_before` | Driver championship points/rank entering the race (shifted, reset each season). |
| `constructor_points_before`, `constructor_position_before` | Constructor championship points/rank entering the race (shifted, reset each season). |
| `podium_rate_prev` | Cumulative rolling driver podium rate strictly over prior career races. |

---

## 5. Dataset Splitting & Preprocessing Experiments

- **Temporal Split:** Train through 2019 (24,148 rows; 12.58% positive); Validation 2020–2021 (780 rows; 15.00% positive); Test 2022–2024 (1,359 rows; 15.01% positive).
- **Preprocessing Order:** Impute then scale (Val PR-AUC 0.7009) vs Scale then impute (0.7003).
- **Imputation Strategy:** Median imputation (Val PR-AUC 0.7009) vs Mean imputation (0.6906).
- **Sensitivity (Indy 500):** Exclude Indy 500 (Val 0.7009 / Test 0.6610; 26,287 rows) vs Include Indy 500 (Val 0.7009 / Test 0.6620; 26,668 rows); difference is negligible (<0.001 PR-AUC).

---

## 6. Predictive Modeling Results

Decision threshold tuned on validation data (0.7584) and evaluated on held-out test.

| Model | Split | PR-AUC | ROC-AUC | F1 | Accuracy | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | Train | 0.5370 | 0.8914 | 0.5595 | 0.8797 | 0.5189 | 0.6070 |
| Logistic regression | Validation | 0.7009 | 0.9181 | 0.6877 | 0.8987 | 0.6397 | 0.7436 |
| Logistic regression | Test | 0.6610 | 0.9246 | 0.6667 | 0.8837 | 0.5852 | 0.7745 |
| Random forest (default) | Train | 0.9754 | 0.9976 | 0.9535 | 0.9878 | 0.9155 | 0.9947 |
| Random forest (default) | Validation | 0.6859 | 0.9128 | 0.6900 | 0.9090 | 0.7054 | 0.6752 |
| Random forest (default) | Test | 0.6480 | 0.9250 | 0.6410 | 0.8904 | 0.6303 | 0.6520 |
| Random forest (regularized) | Train | 0.6277 | 0.9242 | 0.5945 | 0.8967 | 0.5874 | 0.6017 |
| Random forest (regularized) | Validation | 0.7072 | 0.9237 | 0.7043 | 0.9128 | 0.7168 | 0.6923 |
| Random forest (regularized) | Test | 0.6965 | 0.9335 | 0.6420 | 0.8933 | 0.6468 | 0.6373 |
| Shallow FFNN | Train | 0.5799 | 0.9044 | 0.5725 | 0.8879 | 0.5501 | 0.5968 |
| Shallow FFNN | Validation | 0.7433 | 0.9093 | 0.7155 | 0.9154 | 0.7217 | 0.7094 |
| Shallow FFNN | Test | 0.6624 | 0.9264 | 0.6682 | 0.8926 | 0.6229 | 0.7206 |

- Regularized Random Forest achieved the highest held-out test PR-AUC (0.6965).
- Shallow Neural Network achieved the highest validation PR-AUC (0.7433).
- Default Random Forest severely overfitted training data (Train PR-AUC 0.9754 vs Val 0.6859).

---

## 7. Feature-Group Ablation

| Feature Group Removed | Val PR-AUC | Held-Out Test PR-AUC |
|---|---:|---:|
| All Features (Baseline) | 0.7009 | 0.6610 |
| Drop Starting Grid | 0.6749 | 0.6361 |
| Drop Qualifying Session Times/Rank | 0.6909 | 0.6509 |
| Drop Championship Standings | 0.7012 | 0.6565 |
| Drop Rolling Driver Form | 0.7071 | 0.6554 |

- Starting grid is the single most critical feature group; dropping it reduces test PR-AUC by 0.0249.

---

## 8. Model Explainability (XAI)

- **Permutation Importance:** Starting grid dominates feature importance (~0.15 PR-AUC drop), followed by rolling podium form (~0.11) and constructor standings (~0.05).
- **SHAP Beeswarm:** Top grid positions and high prior podium rates strongly elevate predicted log-odds; starting behind grid 10 heavily penalizes podium likelihood.
- **Local LIME Explanation:** For a high-probability validation prediction (Hamilton, prob 85.5%), front-row grid slot (`grid <= -0.85`) contributed +0.36 to podium probability and prior form (`podium_rate_prev > 0.33`) added +0.16.
- **Causal Disclaimer:** Feature importances reflect statistical predictive association within the model, not physical causality.

---

## 9. Inference Demonstration

- **Complete Row:** Hamilton (Grid 2): Predicted probability = 85.46% &rarr; Class 1 (at threshold 0.7584).
- **Partial Row:** Midfield driver with missing qualifying times: Predicted probability = 40.55% &rarr; Class 0.

---

## 10. Limitations

1. **Qualifying Coverage Drift:** Session times are unavailable prior to 1994, requiring imputation.
2. **Constructor Lineage:** Team renamings and acquisitions complicate long-term constructor continuity.
3. **Sprint Races:** Sprint race formats (from 2021) introduce unmodeled race-weekend dynamics.
4. **Starter Status:** Ergast lacks an explicit started flag; pre-race mechanical retirements are inferred from status strings.
