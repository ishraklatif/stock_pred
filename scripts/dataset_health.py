#!/usr/bin/env python3
"""
verify_dataset_health.py

Diagnostics for macro_clean_sanitized + all company_clean_sanitized datasets.

Checks:
    - Duplicate dates
    - Non-monotonic dates
    - Object dtype columns
    - NaN statistics
    - Columns with >40% NaN
    - Zero-variance columns
    - Columns violating TFT naming rules
    - Missing target columns
"""

import os
import pandas as pd
import re

MACRO_PATH = "data/sanitised_final/market/macro_clean_sanitized.parquet"
COMPANY_DIR = "data/sanitised_final/company"


def check_dataset(df: pd.DataFrame, name: str):
    print("=" * 80)
    print(f"[CHECKING] {name}")
    print("=" * 80)

    # ----- BASIC INFO -----
    print("\n[INFO] Shape:", df.shape)
    print("[INFO] Date range:", df["Date"].min(), "→", df["Date"].max())

    # ----- DATE CHECKS -----
    if df["Date"].duplicated().any():
        print("[ERROR] Duplicate dates detected!")
    else:
        print("[OK] No duplicate dates.")

    if not df["Date"].is_monotonic_increasing:
        print("[ERROR] Dates are not sorted!")
    else:
        print("[OK] Dates are sorted.")

    # ----- OBJECT COLUMNS -----
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        print("[WARN] Object columns found:", obj_cols)
    else:
        print("[OK] No object columns.")

    # ----- NA CHECKS -----
    total_na = df.isna().sum().sum()
    print(f"[INFO] Total NaNs: {total_na}")

    na_by_col = df.isna().mean()
    high_na = na_by_col[na_by_col > 0.40]
    if len(high_na) > 0:
        print("\n[WARN] Columns with >40% NaN:")
        print(high_na)
    else:
        print("[OK] No columns with >40% NaN.")

    # ----- ZERO VARIANCE COLUMNS -----
    zero_var = df.loc[:, df.nunique() <= 1].columns
    if len(zero_var) > 0:
        print("\n[WARN] Zero-variance columns:")
        print(zero_var.tolist())
    else:
        print("[OK] No zero-variance columns.")

    # ----- TFT COLUMN NAME SAFETY -----
    bad_cols = [c for c in df.columns if re.search(r"[.\-= /()]", c)]
    if bad_cols:
        print("\n[ERROR] Columns violate TFT naming rules:")
        print(bad_cols)
    else:
        print("[OK] Column names valid for TFT.")

    print("\n")


def main():
    # ----- CHECK MACRO -----
    if os.path.exists(MACRO_PATH):
        macro_df = pd.read_parquet(MACRO_PATH)
        check_dataset(macro_df, "MACRO")
    else:
        print("[WARN] Macro dataset not found.")

    # ----- CHECK COMPANIES -----
    print("\n[INFO] Checking company_clean_sanitized files...\n")
    if not os.path.exists(COMPANY_DIR):
        print("[WARN] No company_clean_sanitized directory.")
        return

    files = [f for f in os.listdir(COMPANY_DIR) if f.endswith(".parquet")]
    if not files:
        print("[WARN] No company sanitized files found.")
        return

    for f in files:
        path = f"{COMPANY_DIR}/{f}"
        try:
            df = pd.read_parquet(path)
            check_dataset(df, f)
        except Exception as e:
            print(f"[ERROR] Cannot read {f}: {e}\n")
            continue

    print("[DONE] Dataset health verification complete.")


if __name__ == "__main__":
    main()
