#!/usr/bin/env python3
"""
merge_level3_into_macro.py

Correct version:
- Merges Level-2 macro dataset with Level-3 parts
- Avoids duplicate merges
- Handles overlapping columns safely

Inputs:
  data/enriched/macro_level2.parquet
  data/level3/rolling_correlations.parquet
  data/level3/rolling_betas.parquet
  data/level3/regime_features.parquet

Output:
  data/final/macro_full.parquet
"""

import os
import pandas as pd

LEVEL2_PATH = "data/level2/macro_level2.parquet"
LEVEL3_DIR  = "data/level3"
OUTPUT_DIR  = "data/level3_merged"
OUTPUT_FILE = f"{OUTPUT_DIR}/macro_full.parquet"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_load(path):
    if os.path.exists(path):
        print(f"[OK] Loaded: {path}")
        return pd.read_parquet(path)
    print(f"[WARN] Missing: {path}")
    return None


def main():
    print("[INFO] Loading Level-2 macro dataset...")
    df = pd.read_parquet(LEVEL2_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Load individual Level-3 components (NOT level3_all)
    corr_df   = safe_load(f"{LEVEL3_DIR}/rolling_correlations.parquet")
    betas_df  = safe_load(f"{LEVEL3_DIR}/rolling_betas.parquet")
    regime_df = safe_load(f"{LEVEL3_DIR}/regime_features.parquet")

    level3_parts = {
        "corr": corr_df,
        "betas": betas_df,
        "regime": regime_df,
    }

    merged = df.copy()

    for name, l3df in level3_parts.items():
        if l3df is None:
            continue

        # Remove duplicate columns before joining
        overlapping_cols = merged.columns.intersection(l3df.columns)
        if len(overlapping_cols) > 0:
            print(f"[WARN] {name} dataset has {len(overlapping_cols)} overlapping columns. Skipping duplicates.")
            l3df = l3df.drop(columns=overlapping_cols)

        print(f"[INFO] Merging Level-3: {name} ({l3df.shape[1]} columns)")
        merged = merged.join(l3df, how="outer")

    # Forward-fill rolling windows
    merged = merged.ffill()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged.to_parquet(OUTPUT_FILE)

    print("\n[OK] Final merged macro dataset saved →", OUTPUT_FILE)
    print("[OK] Final shape:", merged.shape, "\n")


if __name__ == "__main__":
    main()

# python3 scripts/merge_level3_into_macro.py