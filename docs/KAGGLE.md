# Run on Kaggle (step-by-step)

You cannot push from this machine until you add [Kaggle API credentials](https://www.kaggle.com/settings). Until then, use the **web UI** below.

## 1. Package the code zip (once, on your laptop)

```bash
cd "/home/youssef/Documents/Programming/f1-podium-prediction"
bash scripts/package_kaggle_bundle.sh
```

This creates `kaggle/f1-podium-src.zip`.

## 2. Upload the code zip as a Kaggle dataset

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**.
2. Upload `kaggle/f1-podium-src.zip`.
3. Title: **f1-podium-src** (name is flexible; the notebook looks for `f1-podium-src.zip` inside).
4. Set visibility **Public** (required for course submission link).

## 3. Create the notebook

1. [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**.
2. **Add data** (right sidebar):
   - `rohanrao/formula-1-world-championship-1950-2020`
   - Your `f1-podium-src` dataset
3. **File → Upload notebook** → choose `notebooks/milestone1_kaggle.ipynb`  
   (generate it first: `python3 scripts/build_kaggle_notebook.py`)
4. **Settings** → **Internet** → **On**
5. **Save Version** → **Save & Run All** (or **Run All** interactively)

## 4. What you should see

- Setup cell prints `CSV folder: ... | OK: True`
- EDA plots inline
- Metric tables for logistic / RF / FFNN
- SHAP / LIME plots (may take a few minutes on CPU)

Artifacts on Kaggle disk:

- `/kaggle/working/figures/`
- `/kaggle/working/processed/`

Use **Output → Download** after the run.

## 5. Submission link

After a successful public run: **Share** → copy notebook URL for the Google Form.

## Optional: API push (later)

1. Download `kaggle.json` from Kaggle account settings → `~/.kaggle/kaggle.json`
2. `pip install kaggle`
3. Edit `kaggle/kernel-metadata.json` with your slug and run `kaggle kernels push -p kaggle`
