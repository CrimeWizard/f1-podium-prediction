# Getting started — run the project smoothly

Anyone cloning this repo should follow **one** path below. CSV race data is **not** in git; you download it once (~6 MB).

---

## Prerequisites

- Python **3.10+**
- **Wi‑Fi** for first-time `pip install` and data download
- ~2 GB free disk (venv + TensorFlow optional)

---

## Path A — Local (recommended for development)

### 1. Clone and enter the repo

```bash
git clone https://github.com/CrimeWizard/f1-podium-prediction.git
cd f1-podium-prediction
```

### 2. Virtual environment and packages

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -r requirements-ml.txt # needed for NN, SHAP, LIME sections
```

### 3. Download F1 CSVs into `data/raw/`

**Option 1 — kagglehub (no Kaggle API key):**

```bash
pip install kagglehub
python -c "
import kagglehub, shutil
from pathlib import Path
src = Path(kagglehub.dataset_download('rohanrao/formula-1-world-championship-1950-2020'))
dst = Path('data/raw')
dst.mkdir(parents=True, exist_ok=True)
for f in src.glob('*.csv'):
    shutil.copy2(f, dst / f.name)
print('CSV count:', len(list(dst.glob('*.csv'))))
"
```

**Option 2 — Kaggle CLI** (needs `~/.kaggle/kaggle.json`):

```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

**Check:** you should have **14** files in `data/raw/`, including `results.csv`.

### 4. Run the milestone notebook

```bash
jupyter notebook notebooks/milestone1.ipynb
```

In the browser: **Cell → Run All** (first run may take several minutes).

**Success looks like:**

- No red errors through section **§11 Inference**
- `outputs/figures/*.png` created
- `data/processed/model_metrics.csv` exists

### 5. Regenerate the report draft (optional)

```bash
python scripts/generate_report.py
# → updates docs/ANALYTICAL_REPORT.md from latest metrics
```

Export that file to **PDF** for course submission.

### 6. Quick smoke test (no Jupyter)

```bash
python scripts/run_pipeline.py
```

Should print `Loaded tables: [...]` and split sizes.

---

## Path B — Kaggle (course submission notebook)

### 1. Create a new Kaggle notebook

- [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**

### 2. Add data (right sidebar)

| Dataset | Required? |
|---------|-----------|
| [Formula 1 World Championship 1950-2020](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) (rohanrao or Vopani) | **Yes** |
| `f1-podium-src` (your zip) | **No** — git clone is enough |

### 3. Settings

- **Internet → ON** (needed for `git clone` and extra pip packages)

### 4. Fastest verify (one cell)

Paste and run the cell in **[docs/KAGGLE_QUICKFIX.md](docs/KAGGLE_QUICKFIX.md)**.  
You must see `CSV OK: ...` and `Loaded: ['circuits', ...]`.

### 5. Full milestone on Kaggle

- **File → Upload notebook** → `notebooks/milestone1_kaggle.ipynb` from this repo  
  **or** import from GitHub: `CrimeWizard/f1-podium-prediction`
- **Restart session** → **Run All** (run the setup/bootstrap cells first)

More detail: [docs/KAGGLE.md](docs/KAGGLE.md)

### 6. Submission link

**Save version → Save & Run All** (public) → **Share** → copy URL for the Google Form.

---

## Repository map

| Path | What it is |
|------|------------|
| `notebooks/milestone1.ipynb` | Main **Run All** notebook (local) |
| `notebooks/milestone1_kaggle.ipynb` | Same + Kaggle setup cells |
| `src/` | Pipeline code (cleaning, features, models, XAI) |
| `data/raw/` | **You** place Kaggle CSVs here (local) |
| `data/processed/` | Generated parquet/CSV after a run |
| `outputs/figures/` | Plots for report |
| `docs/ANALYTICAL_REPORT.md` | Report draft |
| `docs/REPORT_OUTLINE.md` | Report structure checklist |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Missing CSV tables` / all tables missing | Local: run step 3. Kaggle: add F1 CSV dataset + run bootstrap cell in [KAGGLE_QUICKFIX](docs/KAGGLE_QUICKFIX.md). |
| `No module named 'src'` | Run from repo root; on Kaggle run git-clone cell first. |
| `ModuleNotFoundError: tensorflow` | `pip install -r requirements-ml.txt` |
| Notebook slow on Kaggle | Normal for SHAP/LIME/FFNN on CPU; wait or skip XAI cells for a quick test. |
| `.venv` broken after moving folder | `rm -rf .venv` and recreate from step 2. |

---

## Course submission checklist

1. Public **GitHub** repo link (this repository)  
2. Public **Kaggle notebook** link (after successful Run All)  
3. **PDF report** (from `docs/ANALYTICAL_REPORT.md` + your team’s interpretation)  
4. Form: https://forms.gle/esDxHsvfzffYHVAZ9 — **18 Oct 2026, 11:59 PM**

---

## Regenerate notebooks after editing `scripts/build_notebook.py`

```bash
python scripts/build_notebook.py
python scripts/build_kaggle_notebook.py
```
