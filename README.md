# F1 Podium Prediction (GUC Milestone 1)

Predict whether each driver finishes **on the podium** (`positionOrder ≤ 3`) using only information available **after qualifying and before the race** (no data leakage).

**New to the repo?** → Start here: **[GETTING_STARTED.md](GETTING_STARTED.md)** (local setup, Kaggle, troubleshooting, submission).

## Quick links

| Doc | Purpose |
|-----|---------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | **Step-by-step run guide** for teammates |
| [docs/KAGGLE_QUICKFIX.md](docs/KAGGLE_QUICKFIX.md) | One-cell Kaggle smoke test |
| [docs/KAGGLE.md](docs/KAGGLE.md) | Full Kaggle upload flow |
| [docs/ANALYTICAL_REPORT.md](docs/ANALYTICAL_REPORT.md) | Report draft (export to PDF) |
| [docs/REPORT_OUTLINE.md](docs/REPORT_OUTLINE.md) | Report section checklist |

## One-minute local run

```bash
git clone https://github.com/CrimeWizard/f1-podium-prediction.git
cd f1-podium-prediction
```

Then follow **[GETTING_STARTED.md](GETTING_STARTED.md)** — Path A.

## Submission

- GitHub: this repo (public)
- Kaggle: public notebook after **Run All**
- PDF report + form: https://forms.gle/esDxHsvfzffYHVAZ9 (deadline **18 Oct 2026**)

## Metrics & split

- **Primary metric:** PR-AUC (rare positive class)
- **Split:** train ≤ 2019, validation 2020–2021, test ≥ 2022
