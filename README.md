# F1 Podium Prediction (GUC Milestone 1)

Predict whether each driver finishes **on the podium** (`positionOrder ≤ 3`) using only information available **after qualifying and before the race** (no data leakage).

**New to the repo?** → Start here: **[GETTING_STARTED.md](GETTING_STARTED.md)** (local setup, Kaggle, troubleshooting, submission).

## Course brief

- [Milestone 1: F1 Podium Prediction Companion](docs/course_guidelines/Milestone_1_F1_Podium_Prediction_Companion.pdf)
- [Formula 1 for Newcomers](docs/course_guidelines/F1_for_Newcomers.pdf)

## Quick links

| Doc | Purpose |
|-----|---------|
| [docs/MILESTONE1_SUBMISSION_REPORT.pdf](docs/MILESTONE1_SUBMISSION_REPORT.pdf) | Previous report from Kaggle Version 5; regenerate after rerunning the corrected standings features |
| [docs/MILESTONE1_SUBMISSION_REPORT.md](docs/MILESTONE1_SUBMISSION_REPORT.md) | Source for the Version 5 report; refresh metrics and figures after the corrected run |
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
- Kaggle: https://www.kaggle.com/code/crimewizard/acl-ms1 (Version 5 completed; save a new Run All version after publishing the current standings fix)
- PDF report: [docs/MILESTONE1_SUBMISSION_REPORT.pdf](docs/MILESTONE1_SUBMISSION_REPORT.pdf)
- Submission Form: https://forms.gle/esDxHsvfzffYHVAZ9 (course extension communicated by the team: **19 Oct 2026, 11:59 PM Cairo time**; this deadline passed on 20 Oct 2026, so confirm late-submission options with course staff)

## Metrics & split

- **Primary metric:** PR-AUC (rare positive class)
- **Split:** train ≤ 2019, validation 2020–2021, test ≥ 2022
