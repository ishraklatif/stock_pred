#!/usr/bin/env python3
"""
merge_company_with_macro_final.py

Merges:
    - macro_full_level4 (Levels 1–4)
    - processed/enriched company files

Output:
    data/company_final/<TICKER>_merged.parquet

This ensures each company gets:
    - company OHLCV + indicators
    - macro Level 1
    - macro Level 2
    - Level 3 rolling betas/correlations
    - Level 4 sentiment
"""

import os
import pandas as pd
from glob import glob

MACRO_PATH = "data/level4_final/market/macro_clean.parquet"
COMPANY_DIR = "data/processed_companies"   # your 001_ processed + enriched output
OUT_DIR = "data/level4_company"

os.makedirs(OUT_DIR, exist_ok=True)

def sanitize_ticker(name):
    # make name folder-friendly
    return (
        name.replace(".parquet", "")
            .replace(".", "_")
            .replace("^", "_")
            .replace("-", "_")
            .upper()
    )


def main():
    print("[INFO] Loading full macro dataset with Levels 1–4...")
    macro = pd.read_parquet(MACRO_PATH)
    macro = macro.sort_index()

    print("[INFO] Searching for enriched company files...")
    company_files = sorted(glob(f"{COMPANY_DIR}/*.parquet"))

    if len(company_files) == 0:
        print("[ERROR] No company files found in:", COMPANY_DIR)
        return

    print(f"[INFO] Found {len(company_files)} company enriched files.")

    for path in company_files:
        fname = os.path.basename(path)
        symbol = sanitize_ticker(fname)

        print("\n===============================================")
        print(f"[MERGING] {symbol}")
        print("===============================================")

        df_c = pd.read_parquet(path)
        df_c = df_c.sort_index()

        # merge on Date index
        merged = macro.join(df_c, how="left")

        # forward/back fill only company columns
        company_cols = [c for c in df_c.columns if c != "Date"]
        merged[company_cols] = merged[company_cols].ffill().bfill()

        out_path = f"{OUT_DIR}/{symbol}_merged.parquet"
        merged.to_parquet(out_path)

        print(f"[OK] Saved: {out_path}")
        print(f"[OK] Shape: {merged.shape}")

    print("\n[DONE] All companies merged with Level 1–4 macro data.\n")


if __name__ == "__main__":
    main()

# python3 scripts/merge_company_with_macro.py
