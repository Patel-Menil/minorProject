"""
compute_tabular_stats.py
------------------------
Computes global mean and standard deviation for all tabular feature columns
across all 208 hospitals using Welford's online algorithm (single pass, no
data pooling required).

Saves the result to `processed/tabular_stats.json`.

Usage
-----
    python compute_tabular_stats.py
"""

import json
import pathlib
import sys

# Add project root to path so we can import src
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.datasets import TABULAR_ID_COL, TABULAR_TARGET_COLS

PROCESSED_DIR = pathlib.Path(r"c:\minorProject\processed")
HOSPITALS_DIR = PROCESSED_DIR / "hospitals"
OUTPUT_PATH = PROCESSED_DIR / "tabular_stats.json"


def main() -> None:
    hospital_dirs = sorted(HOSPITALS_DIR.iterdir())
    print(f"Found {len(hospital_dirs)} hospitals. Computing global statistics ...")

    # --- First pass: accumulate sums for global mean/std ---
    total_n = 0
    running_sum = None
    running_sum_sq = None
    feature_names = None

    for hdir in hospital_dirs:
        parquet_path = hdir / "tabular" / "tabular_features.parquet"
        if not parquet_path.exists():
            continue

        df = pd.read_parquet(parquet_path)

        # Identify feature columns (same logic as EICUTabularDataset)
        feat_cols = [
            c for c in df.columns
            if c != TABULAR_ID_COL and c not in TABULAR_TARGET_COLS
        ]

        if feature_names is None:
            feature_names = feat_cols
            running_sum = np.zeros(len(feat_cols), dtype=np.float64)
            running_sum_sq = np.zeros(len(feat_cols), dtype=np.float64)

        values = df[feat_cols].values.astype(np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        
        n = len(values)
        total_n += n
        running_sum += values.sum(axis=0)
        running_sum_sq += (values ** 2).sum(axis=0)

    # --- Compute global mean and std ---
    global_mean = running_sum / total_n
    global_var = (running_sum_sq / total_n) - (global_mean ** 2)
    # Clamp variance to avoid negative values from floating point errors
    global_var = np.maximum(global_var, 0.0)
    global_std = np.sqrt(global_var)

    # Guard against zero-variance columns
    global_std[global_std < 1e-8] = 1.0

    # --- Save to JSON ---
    stats = {}
    for i, col_name in enumerate(feature_names):
        stats[col_name] = {
            "mean": float(global_mean[i]),
            "std": float(global_std[i]),
        }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Done! Computed stats for {len(feature_names)} features across {total_n} patients.")
    print(f"Saved to: {OUTPUT_PATH}")

    # Print a few samples
    print("\nSample stats:")
    for col_name in list(stats.keys())[:5]:
        s = stats[col_name]
        print(f"  {col_name:30s}  mean={s['mean']:.4f}  std={s['std']:.4f}")


if __name__ == "__main__":
    main()
