# Kaggle quick fix (if nothing works)

## Do this exactly

1. **New notebook** on Kaggle (fresh start).
2. **Add data** → search **Formula 1 World Championship** (Vopani or rohanrao) → Add.  
   You do **not** need `f1-podium-src` if Internet is on.
3. **Settings** → **Internet → ON**.
4. **Delete every cell** in the notebook.
5. Paste **one** code cell from below → **Run**.
6. If it prints `Loaded tables: [...]` → success. Then upload `notebooks/milestone1.ipynb` from GitHub and Run All, **or** keep using the test cell.

## One-cell smoke test

```python
import os, sys, subprocess
from pathlib import Path

# Find F1 CSVs
for r in Path("/kaggle/input").rglob("results.csv"):
    d = r.parent
    if (d / "circuits.csv").exists():
        os.environ["F1_DATA_RAW"] = str(d)
        print("CSV OK:", d)
        break
else:
    raise SystemExit("STOP: Add Formula 1 World Championship dataset (Add data).")

# Clone code
repo = Path("/kaggle/working/f1-podium-prediction")
if not (repo / "src/io.py").exists():
    subprocess.run(["git", "clone", "--depth", "1",
        "https://github.com/CrimeWizard/f1-podium-prediction.git", str(repo)], check=True)
sys.path.insert(0, str(repo))
os.chdir(repo)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn", "shap", "lime"], check=False)

from src.io import load_raw_tables
tables = load_raw_tables()
print("Loaded tables:", list(tables.keys()))
```

## Common mistakes

| Mistake | Symptom |
|---------|---------|
| Only `f1-podium-src` added | `No module named src` or no CSVs |
| No F1 CSV dataset | All tables missing |
| Internet OFF | `git clone` fails |
| Ran §0 before imports cell on old notebook | Wrong paths — **Restart session**, **Run All** top to bottom |
