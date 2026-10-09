#!/usr/bin/env python3
"""Run full milestone pipeline (for Kaggle: python scripts/run_pipeline.py)."""
from src.kaggle_bootstrap import bootstrap_kaggle
from src.io import load_raw_tables, save_processed
from src.cleaning import build_base_results_table
from src.features import build_modeling_table
from src.splits import temporal_split

print("Bootstrap:", bootstrap_kaggle())
tables = load_raw_tables()
print("Loaded tables:", list(tables.keys()))
base, dups = build_base_results_table(tables)
print("Base rows:", len(base), "duplicate audit:", len(dups))
model_df = build_modeling_table(tables, base)
save_processed(model_df, "modeling_table")
train, val, test = temporal_split(model_df)
print("Split sizes:", len(train), len(val), len(test))
print("OK — open notebooks/milestone1.ipynb for full EDA/models or continue in notebook.")
