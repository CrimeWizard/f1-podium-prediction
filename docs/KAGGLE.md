# Update and run the Milestone 1 notebook on Kaggle

## Rerun the existing submission notebook

The update changes both project code and notebook cells. The setup cell fetches current project code from GitHub `main`, but it cannot replace cells already saved in Kaggle. Import the latest generated notebook file from this repository so the full analysis table is used for Q1–Q3 while the modeling table continues to exclude the Indianapolis 500. The attached Formula 1 dataset stays as the notebook input; a separate `f1-podium-src` dataset is not needed.

## Kaggle steps

1. Download `notebooks/milestone1_kaggle.ipynb` from the repository's `main` branch. Use GitHub's file page **Download raw file** action to save the `.ipynb` file; do not open the raw JSON page as a notebook.
2. Open [ACL_MS1](https://www.kaggle.com/code/crimewizard/acl-ms1) and sign in to the owner account. Use **File → Import Notebook** to import the downloaded `.ipynb`. If Kaggle creates a fork, use its new URL for submission.
3. In the **Input** panel, confirm **Formula 1 World Championship (1950–2024)** is attached.
4. In **Settings**, turn **Internet** on. The first setup cell fetches project code from GitHub `main`.
5. Choose **Run → Run All**. Wait for every cell to finish; resolve any red error cell before proceeding.
6. Confirm setup prints **Source revision: `34b2186` or newer**, plus the CSV path and `OK: True`.
7. Confirm the outputs include key audits; Q1–Q3 tables and plots; four-model ROC-AUC/PR-AUC/F1 metrics; preprocessing and ablation tables; permutation, SHAP, and local explanation outputs; and both inference examples.
8. Choose **Save Version → Save & Run All**. Verify the saved version is public and the run status is successful.
9. From **Output**, download the new notebook with outputs and the `figures/` and `processed/` artifacts. Use those corrected outputs to regenerate and replace the PDF report.

## What a successful run should show

- Setup prints the Kaggle CSV folder and `OK: True`.
- All 14 expected tables are loaded and the key audit covers them.
- The Q1 chart has four panels (qualifying/grid × pre/from 2014); rankings use circuits with at least 20 starts in the relevant group, while full denominators are saved.
- Q2 bins are `1–5`, `6–10`, `11–15`, and `16+`; missing/zero grid and unknown-country rows are reported as excluded.
- Q3 reports actual race starters, excluding explicit non-start statuses and retaining pit-lane starts where status indicates a start.
- Model metrics include logistic regression, default random forest, shallow FFNN, and regularized random forest, each on train/validation/test.

The Google Form deadline previously shown by the course team was **19 Oct 2026, 11:59 PM Cairo time**. That date has passed as of 20 Oct 2026; check the course announcement or contact staff about late submission/reopening.
