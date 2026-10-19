# Update and run the Milestone 1 notebook on Kaggle

## Before opening Kaggle

Use the generated notebook `notebooks/milestone1_kaggle.ipynb` from the latest GitHub `main` branch. It contains the corrected Q1–Q3 analysis and the complete evaluation outputs. The source code is cloned from GitHub by the notebook; do not add a separate `f1-podium-src` dataset.

## Kaggle steps

1. Open [the current ACL_MS1 notebook](https://www.kaggle.com/code/crimewizard/acl-ms1) and sign in to the **CrimeWizard owner account**.
2. If Kaggle shows **Copy & Edit**, click it to create an editable copy. If you see **Edit**, open the existing notebook.
   - **Copy & Edit creates a fork.** If the URL changes, use that new notebook URL in the submission form; the old link will remain Version 2.
3. In the notebook editor, use **File → Import Notebook** and select the updated local file `notebooks/milestone1_kaggle.ipynb`.
4. In the right **Input** panel, confirm **Formula 1 World Championship (1950–2024)** is attached. No other dataset is needed.
5. Open **Settings** and turn **Internet** on. The setup cell clones the public GitHub repository and installs optional SHAP/LIME packages.
6. Choose **Run → Run All**. Wait for every cell to finish. Resolve any red error cell before proceeding.
7. Confirm the output includes: primary-key and foreign-key audits; Q1 qualifying-versus-grid comparisons; Q2 home/away rates by grid bin; Q3 mechanical-retirement rates by era; the four-model ROC-AUC/PR-AUC/F1 table; preprocessing and ablation tables; permutation, SHAP, and local explanation outputs; and the inference examples.
8. Choose **Save Version → Save & Run All**. Verify the saved version is public and the run status is successful. The notebook link alone does not prove this updated run is saved.
9. From **Output**, download the new `figures/` and `processed/` artifacts. Use those outputs to regenerate the PDF report and replace the old report in the repository and submission form.

## What a successful run should show

- Setup prints the Kaggle CSV folder and `OK: True`.
- All 14 expected tables are loaded and the key audit covers them.
- The Q1 chart has four panels (qualifying/grid × pre/from 2014); rankings use circuits with at least 20 starts in the relevant group, while full denominators are saved.
- Q2 bins are `1–5`, `6–10`, `11–15`, and `16+`; missing/zero grid and unknown-country rows are reported as excluded.
- Q3 reports actual race starters, excluding explicit non-start statuses and retaining pit-lane starts where status indicates a start.
- Model metrics include logistic regression, default random forest, shallow FFNN, and regularized random forest, each on train/validation/test.

The Google Form deadline previously shown by the course team was **19 Oct 2026, 11:59 PM Cairo time**. That date has passed as of 20 Oct 2026; check the course announcement or contact staff about late submission/reopening.
