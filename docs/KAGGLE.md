# Update and run the Milestone 1 notebook on Kaggle

## Rerun the existing submission notebook

The public `ACL_MS1` notebook has a setup cell that fetches the latest project code from GitHub `main`. For a normal update, keep using the existing Kaggle notebook; do not import the raw GitHub `.ipynb` page. The attached Formula 1 dataset stays as the notebook input, and no separate `f1-podium-src` dataset is needed.

## Kaggle steps

1. Open [ACL_MS1](https://www.kaggle.com/code/crimewizard/acl-ms1) and sign in to the owner account. Click **Edit** to edit this notebook. If Kaggle requires **Copy & Edit**, make a fork and use that notebook's URL for submission.
2. In the **Input** panel, confirm **Formula 1 World Championship (1950–2024)** is attached.
3. In **Settings**, turn **Internet** on. The first setup cell fetches GitHub `main`, so it will pick up the season-boundary correction.
4. Choose **Run → Run All**. Wait for every cell to finish; resolve any red error cell before proceeding.
5. Confirm setup prints **Source revision: `34b2186` or newer**, plus the CSV path and `OK: True`.
6. Confirm the outputs include key audits; Q1–Q3 tables and plots; four-model ROC-AUC/PR-AUC/F1 metrics; preprocessing and ablation tables; permutation, SHAP, and local explanation outputs; and both inference examples.
7. Choose **Save Version → Save & Run All**. Verify the saved version is public and the run status is successful.
8. From **Output**, download the new notebook with outputs and the `figures/` and `processed/` artifacts. Use those corrected outputs to regenerate and replace the PDF report.

## What a successful run should show

- Setup prints the Kaggle CSV folder and `OK: True`.
- All 14 expected tables are loaded and the key audit covers them.
- The Q1 chart has four panels (qualifying/grid × pre/from 2014); rankings use circuits with at least 20 starts in the relevant group, while full denominators are saved.
- Q2 bins are `1–5`, `6–10`, `11–15`, and `16+`; missing/zero grid and unknown-country rows are reported as excluded.
- Q3 reports actual race starters, excluding explicit non-start statuses and retaining pit-lane starts where status indicates a start.
- Model metrics include logistic regression, default random forest, shallow FFNN, and regularized random forest, each on train/validation/test.

The Google Form deadline previously shown by the course team was **19 Oct 2026, 11:59 PM Cairo time**. That date has passed as of 20 Oct 2026; check the course announcement or contact staff about late submission/reopening.
