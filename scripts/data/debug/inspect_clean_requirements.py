#!/usr/bin/env python3
"""
inspect_clean_requirements.py

Scan raw macro, market and company folders and detect:
- Missing OHLC columns
- Rows where *all* price columns are NaN
- Object columns
- Rows where date is invalid
- Number of NaNs per column

This tells us exactly what cleaning is REQUIRED (and what not to do).
"""

import os
import pandas as pd
import yaml

CONFIG = "config/data.yaml"

def load_cfg():
    with open(CONFIG, "r") as f:
        return yaml.safe_load(f)

def inspect_folder(title, folder):
    print(f"\n==============================")
    print(f" INSPECTION: {title}")
    print(f"==============================")

    files = [f for f in os.listdir(folder) if f.endswith(".parquet")]
    for f in files:
        path = os.path.join(folder, f)
        df = pd.read_parquet(path)

        print(f"\n--- {f} ---")
        print("Shape:", df.shape)

        # object columns
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            print("Object columns:", obj_cols)

        # check OHLC existence
        missing = [c for c in ["open","high","low","close"] if c not in df.columns]
        if missing:
            print("Missing OHLC cols:", missing)

        # rows with all OHLC missing
        if all(c in df.columns for c in ["open","high","low","close"]):
            mask = df[["open","high","low","close"]].isna().all(axis=1)
            if mask.any():
                print("Rows with ALL OHLC missing:", mask.sum())

        # invalid dates
        if df["date"].isna().any():
            print("Rows with invalid date:", df["date"].isna().sum())

        # basic NaN stats
        nan_stats = df.isna().sum()
        print("Top NaN columns:")
        print(nan_stats[nan_stats > 0].sort_values(ascending=False).head(10))


def main():
    cfg = load_cfg()

    inspect_folder("MACRO", cfg["data"]["sources"]["macro_folder"])
    inspect_folder("MARKET", cfg["data"]["sources"]["market_folder"])
    inspect_folder("COMPANIES", cfg["data"]["sources"]["companies_folder"])

if __name__ == "__main__":
    main()
