#!/usr/bin/env python3
"""
inspect_one_company_features_compact.py

Compact version:
 - Prints all columns in ONE LINE
 - Prints each category group in ONE LINE
 - Avoids terminal spam
"""

import os
import pandas as pd


# ---------------------------------------------------------
# CHANGE THIS PATH to the company file you want to inspect
# ---------------------------------------------------------
COMPANY_FILE = "data/tft_ready/AGL.AX.parquet"


def print_line(title, items):
    """Utility: print features in a single line."""
    print(f"{title} ({len(items)}):")
    if len(items) == 0:
        print("  (none)\n")
        return
    print("  " + ", ".join(items) + "\n")


def main():
    print("\n====================================================")
    print("         SINGLE COMPANY FEATURE SCAN (COMPACT)")
    print("====================================================\n")

    if not os.path.exists(COMPANY_FILE):
        raise FileNotFoundError(f"[ERROR] File not found: {COMPANY_FILE}")

    print(f"[INFO] Loading: {COMPANY_FILE}\n")
    df = pd.read_parquet(COMPANY_FILE)

    # ---------------- BASIC ----------------
    print(f"Shape: {df.shape}\n")

    # ---------------- FULL COLUMNS ----------------
    print_line("ALL COLUMNS", df.columns.tolist())

    # ---------------- TYPES ----------------
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols     = [c for c in df.columns if df[c].dtype == object]
    dt_cols      = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    print_line("NUMERIC FEATURES", numeric_cols)
    print_line("CATEGORICAL FEATURES", cat_cols)
    print_line("DATETIME FEATURES", dt_cols)

    # ---------------- CATEGORY GROUPS ----------------
    calendar_keywords = ["day", "week", "month", "quarter", "holiday", "is_"]
    tech_keywords     = ["sma", "ema", "macd", "rsi", "atr", "vol", "bb", "adx"]
    macro_keywords    = ["axjo", "gspc", "ftse", "n225", "hsi", "vix", "000001", "000300"]
    market_keywords   = ["gold", "oil", "aud", "usd", "copper", "silver", "dbc"]
    sent_keywords     = ["sent", "news"]

    calendar_cols = [c for c in df.columns if any(k in c.lower() for k in calendar_keywords)]
    tech_cols     = [c for c in df.columns if any(k in c.lower() for k in tech_keywords)]
    macro_cols    = [c for c in df.columns if any(k in c.lower() for k in macro_keywords)]
    market_cols   = [c for c in df.columns if any(k in c.lower() for k in market_keywords)]
    sent_cols     = [c for c in df.columns if any(k in c.lower() for k in sent_keywords)]

    print_line("CALENDAR FEATURES", calendar_cols)
    print_line("TECHNICAL INDICATORS", tech_cols)
    print_line("MACRO FEATURES", macro_cols)
    print_line("MARKET FEATURES", market_cols)
    print_line("NEWS SENTIMENT FEATURES", sent_cols)

    # ---------------- MISSING VALUE SUMMARY ----------------
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print_line("FEATURES WITH MISSING VALUES", [f"{k}({v})" for k, v in missing.items()])

    print("====================================================")
    print("             FEATURE SCAN COMPLETE")
    print("====================================================\n")


if __name__ == "__main__":
    main()
