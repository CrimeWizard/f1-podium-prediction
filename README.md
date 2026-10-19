# F1 Podium Prediction (GUC Milestone 1)

Predict whether each driver finishes **on the podium** (`positionOrder ≤ 3`) using only information available **after qualifying and before the race** (no data leakage).

**New to the repo?** → Start here: **[GETTING_STARTED.md](GETTING_STARTED.md)** (local setup, Kaggle, troubleshooting, submission).

## Course brief

- [Milestone 1: F1 Podium Prediction Companion](docs/course_guidelines/Milestone_1_F1_Podium_Prediction_Companion.pdf)
- [Formula 1 for Newcomers](docs/course_guidelines/F1_for_Newcomers.pdf)

## Quick links

| Doc | Purpose |
|-----|---------|
| [docs/MILESTONE1_SUBMISSION_REPORT.pdf](docs/MILESTONE1_SUBMISSION_REPORT.pdf) | Final report from successful Kaggle Version 7 run |
| [docs/MILESTONE1_SUBMISSION_REPORT.md](docs/MILESTONE1_SUBMISSION_REPORT.md) | Source for the final report |
| [GETTING_STARTED.md](GETTING_STARTED.md) | **Step-by-step run guide** for teammates |
| [docs/KAGGLE_QUICKFIX.md](docs/KAGGLE_QUICKFIX.md) | One-cell Kaggle smoke test |
| [docs/KAGGLE.md](docs/KAGGLE.md) | Full Kaggle upload flow |
| [docs/ANALYTICAL_REPORT.md](docs/ANALYTICAL_REPORT.md) | Report draft skeleton |
| [docs/REPORT_OUTLINE.md](docs/REPORT_OUTLINE.md) | Report section checklist |

## One-minute local run

```bash
git clone https://github.com/CrimeWizard/f1-podium-prediction.git
cd f1-podium-prediction
```

Then follow **[GETTING_STARTED.md](GETTING_STARTED.md)** — Path A.

## Submission

- GitHub: https://github.com/CrimeWizard/f1-podium-prediction (public)
- Kaggle: https://www.kaggle.com/code/crimewizard/acl-ms1 (Version 7 completed successfully (113.1 seconds; 110 output files); source revision 3f5da97)
- PDF report: regenerated from the successful Kaggle Version 7 export.
- Submission form: https://forms.gle/esDxHsvfzffYHVAZ9

## Metrics & split

- **Primary metric:** PR-AUC (rare positive class)
- **Split:** train ≤ 2019, validation 2020–2021, test ≥ 2022
