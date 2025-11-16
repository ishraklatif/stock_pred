#!/usr/bin/env python3
"""
clean_macro_dataset.py

Cleans the merged macro dataset (Level 1–4 signals).
This produces the final feature matrix for TFT and all ML models.

Steps:
1. Load macro_full_level4.parquet
2. Drop rows before 2007
3. Remove object/string columns (symbol, metadata)
4. Drop columns with >50% NaNs
5. Drop rows with >30% NaNs
6. Forward-fill + backward-fill remaining NaNs
7. Convert all dtypes to float32
8. Save final cleaned dataset

Output:
    data/final/macro_clean.parquet
"""

import os
import pandas as pd
import numpy as np

INPUT_PATH  = "data/level4_merged/macro_full_level4.parquet"
OUTPUT_PATH = "data/level4_final/market/macro_clean.parquet"


def main():
    print("[INFO] Loading dataset...")
    df = pd.read_parquet(INPUT_PATH)

    # --- FIX: Ensure Date exists as a NORMAL COLUMN ---
    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            print("[FIX] Date is index. Resetting index...")
            df = df.reset_index()
        else:
            raise ValueError("No Date column or datetime index found.")

    # Convert to datetime & sort
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Remove early years with extreme NaN rates
    print("[INFO] Dropping rows before 2007...")
    df = df[df["Date"] >= "2007-01-01"]

    print("[INFO] Remaining rows after date filter:", len(df))

    # Drop object columns (symbols / metadata)
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        print("[INFO] Dropping object columns:", object_cols)
        df = df.drop(columns=object_cols)

    # Drop columns with >50% NaN
    col_thresh = int(len(df) * 0.50)
    before_cols = df.shape[1]
    df = df.dropna(axis=1, thresh=col_thresh)
    after_cols = df.shape[1]
    print(f"[INFO] Dropped {before_cols - after_cols} columns (>50% NaN)")

    # Drop rows with >30% NaN
    row_thresh = int(df.shape[1] * 0.70)
    before_rows = df.shape[0]
    df = df.dropna(axis=0, thresh=row_thresh)
    after_rows = df.shape[0]
    print(f"[INFO] Dropped {before_rows - after_rows} rows (>30% NaN)")

    # Forward/back fill
    print("[INFO] Forward-filling and back-filling NaNs...")
    df = df.ffill().bfill()

    # Convert to float32
    print("[INFO] Converting numeric columns to float32...")
    for col in df.columns:
        if col != "Date":
            df[col] = df[col].astype("float32")

    # Save cleaned dataset
    print("[INFO] Saving cleaned dataset...")
    df.to_parquet(OUTPUT_PATH)

    print("[OK] Cleaned dataset saved →", OUTPUT_PATH)
    print("[OK] Final shape:", df.shape)


if __name__ == "__main__":
    main()
