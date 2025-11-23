#!/usr/bin/env python3
"""
deep_feature_scan.py

Full deep-diagnostic scanner for your TFT multiseries dataset.
- detects bad columns, NaN issues, inf issues
- detects constant & zero-variance features
- detects duplicated columns
- detects very high correlations (redundant)
- ensures time_idx is correct per series
- prints all column groups in one-line compact format
"""

import os
import pandas as pd
import numpy as np

FILE = "data/tft_ready_multiseries/train.parquet"


# -----------------------------------------------
# UTIL: print in one line
# -----------------------------------------------
def print_line(title, items):
    print(f"{title} ({len(items)}):")
    if not items:
        print("  (none)\n")
        return
    print("  " + ", ".join(map(str, items)) + "\n")


def main():
    print("\n====================================================")
    print("               DEEP FEATURE SCAN v2")
    print("====================================================\n")

    if not os.path.exists(FILE):
        raise FileNotFoundError(f"[ERROR] File not found: {FILE}")

    print(f"[INFO] Loading dataset: {FILE}\n")
    df = pd.read_parquet(FILE)

    # ---------------- BASIC ----------------
    print(f"Shape: {df.shape}\n")

    # ---------------- ALL COLUMNS ----------------
    all_cols = df.columns.tolist()
    print_line("ALL COLUMNS", all_cols)

    # ---------------- COLUMN TYPES ----------------
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    dt_cols  = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    print_line("NUMERIC", num_cols)
    print_line("CATEGORICAL", cat_cols)
    print_line("DATETIME", dt_cols)

    # ------------------------------------------------------------
    # GROUP COLUMNS BY PATTERN
    # ------------------------------------------------------------
    calendar_keywords = ["day", "week", "month", "quarter", "holiday", "year", "is_"]
    tech_keywords     = ["ema", "rsi", "macd", "bb_", "vol_", "ret_", "return", "log_return", "obv", "candle"]
    macro_keywords    = ["axjo", "gspc", "ftse", "n225", "hsi", "000001", "000300", "vix"]
    market_keywords   = ["aud", "usd", "gold", "oil", "copper", "silver", "dbc", "gc_f", "cl_f", "si_f", "hg_f", "bz_f"]
    sent_keywords     = ["sent_", "_pos", "_neg", "_neu"]

    cal_cols  = [c for c in df.columns if any(k in c.lower() for k in calendar_keywords)]
    tech_cols = [c for c in df.columns if any(k in c.lower() for k in tech_keywords)]
    macro_cols = [c for c in df.columns if any(k in c.lower() for k in macro_keywords)]
    market_cols = [c for c in df.columns if any(k in c.lower() for k in market_keywords)]
    sent_cols  = [c for c in df.columns if any(k in c.lower() for k in sent_keywords)]

    print_line("CALENDAR FEATURES", cal_cols)
    print_line("TECHNICAL FEATURES", tech_cols)
    print_line("MACRO FEATURES", macro_cols)
    print_line("MARKET FEATURES", market_cols)
    print_line("NEWS SENTIMENT", sent_cols)

    # ------------------------------------------------------------
    # MISSING VALUE SUMMARY
    # ------------------------------------------------------------
    na = df.isna().sum()
    na = na[na > 0].sort_values(ascending=False)
    print_line("COLUMNS WITH NaN", [f"{col}={na[col]}" for col in na.index])

    # ------------------------------------------------------------
    # INF VALUES
    # ------------------------------------------------------------
    inf_cols = []
    for c in df.columns:
        # Only attempt on numeric columns
        if pd.api.types.is_numeric_dtype(df[c]):
            if np.isinf(df[c].astype(float)).any():
                inf_cols.append(c)
    print_line("COLUMNS WITH INF", inf_cols)


    # ------------------------------------------------------------
    # CONSTANT COLUMNS (useless)
    # ------------------------------------------------------------
    constant_cols = [c for c in num_cols if df[c].nunique() <= 1]
    print_line("CONSTANT FEATURES", constant_cols)

    # ------------------------------------------------------------
    # ZERO VARIANCE / LOW VARIANCE (potentially useless)
    # ------------------------------------------------------------
    low_var_cols = []
    for c in num_cols:
        if df[c].std() == 0:
            low_var_cols.append(c)
    print_line("ZERO-VARIANCE FEATURES", low_var_cols)

    # ------------------------------------------------------------
    # DUPLICATE COLUMNS (same exact values)
    # ------------------------------------------------------------
    duplicates = []
    seen = {}
    for c in df.columns:
        series_tuple = tuple(df[c].values)
        if series_tuple in seen:
            duplicates.append((c, seen[series_tuple]))
        else:
            seen[series_tuple] = c
    print_line("DUPLICATED COLUMNS", [f"{a} == {b}" for a, b in duplicates])

    # ------------------------------------------------------------
    # HIGH CORRELATION (> 0.98) — redundant columns
    # ------------------------------------------------------------
    print("[INFO] Computing correlations (this may take a moment)...")
    corr = df[num_cols].corr().abs()
    high_corr_pairs = []
    for i, col1 in enumerate(num_cols):
        for col2 in num_cols[i+1:]:
            if corr.loc[col1, col2] > 0.98:
                high_corr_pairs.append(f"{col1} ~ {col2} ({corr.loc[col1,col2]:.3f})")

    print_line("HIGH-CORRELATION FEATURES (>0.98)", high_corr_pairs)

    # ------------------------------------------------------------
    # TIME INDEX CHECKS
    # ------------------------------------------------------------
    if "time_idx" in df.columns and "series" in df.columns:
        invalid_series = []
        for s, g in df.groupby("series"):
            if not g["time_idx"].is_monotonic_increasing:
                invalid_series.append(s)
        print_line("SERIES WITH INVALID time_idx ORDER", invalid_series)

    # ------------------------------------------------------------
    # DUPLICATE DATE CHECK (per series)
    # ------------------------------------------------------------
    if "Date" in df.columns:
        dup_date_series = []
        for s, g in df.groupby("series"):
            if g["Date"].duplicated().any():
                dup_date_series.append(s)
        print_line("SERIES WITH DUPLICATE DATES", dup_date_series)

    print("====================================================")
    print("           DEEP FEATURE SCAN COMPLETE")
    print("====================================================\n")


if __name__ == "__main__":
    main()
