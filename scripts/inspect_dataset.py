#!/usr/bin/env python3
"""
inspect_dataset.py

Tool to inspect any merged/enriched dataset in StockPred.
Provides:
- Shape, columns, dtype summary
- Date continuity check
- Missing-values analysis
- Missing-values heatmap
- Correlation heatmap (safe mode)
- Column grouping by pattern
- Basic preview plots

Usage:
    python3 scripts/inspect_dataset.py --file data/enriched/macro_level2.parquet
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")
    print(f"[INFO] Loading dataset: {path}")
    df = pd.read_parquet(path)
    return df


def ensure_date(df):
    """Ensure dataset has a Date column as datetime."""
    # Move index to column if needed
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    date_cols = [c for c in df.columns if str(c).lower() in ("date", "datetime")]

    if len(date_cols) == 0:
        raise RuntimeError("No Date column found!")

    df = df.rename(columns={date_cols[-1]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    return df


# ------------------------------------------
# INSPECTION CORE
# ------------------------------------------

def inspect_basic(df):
    print("\n===== BASIC DATASET INFO =====")
    print(df.info())

    print("\n===== SHAPE =====")
    print(df.shape)

    print("\n===== DATE RANGE =====")
    print(df["Date"].min(), "→", df["Date"].max())

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== LAST 5 ROWS =====")
    print(df.tail())


def inspect_column_groups(df):
    print("\n===== COLUMN GROUPS =====")

    groups = {
        "Calendar": [c for c in df.columns if any(k in c.lower() for k in [
            "day_of_week", "month", "quarter", "is_month_end", "is_us_holiday"
        ])],
        "VIX/DXY/Yields": [c for c in df.columns if any(k in c for k in [
            "VIX", "DXY", "US10Y", "US2Y"
        ])],
        "Indices": [c for c in df.columns if any(k in c for k in [
            "AXJO", "SPY", "FTSE", "N225", "HSI", "SSE", "CSI", "NIFTY", "KS11", "TWII"
        ])],
        "FX": [c for c in df.columns if "AUD" in c],
        "Commodities": [c for c in df.columns if any(k in c for k in [
            "GC", "BZ", "TIO", "COPPER", "DBC", "COMMODITY"
        ])],
        "Company": [c for c in df.columns if c.endswith(".AX") or "AX_" in c],
    }

    for group_name, cols in groups.items():
        print(f"\n{group_name} ({len(cols)} columns):")
        print(cols[:15], "..." if len(cols) > 15 else "")


def inspect_missing_values(df):
    print("\n===== MISSING VALUES SUMMARY =====")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.head(20))

    # Heatmap (optional)
    try:
        print("\n[INFO] Generating missing-values heatmap...")
        plt.figure(figsize=(14, 6))
        sns.heatmap(df.isna(), cbar=False)
        plt.title("Missing Values Heatmap")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[WARN] Could not generate missing-values heatmap: {e}")


def check_date_continuity(df):
    print("\n===== DATE CONTINUITY CHECK =====")

    df = df.sort_values("Date")
    df["date_diff"] = df["Date"].diff().dt.days

    gaps = df["date_diff"].value_counts().sort_index()
    print("Gap distribution:")
    print(gaps)

    # Plot gaps
    plt.figure(figsize=(10, 4))
    gaps.plot(kind="bar")
    plt.title("Date Gap Distribution (days between rows)")
    plt.xlabel("Gap size (days)")
    plt.ylabel("Frequency")
    plt.show()


def inspect_correlations(df):
    print("\n===== CORRELATION MATRIX (SAFE MODE) =====")

    # Use only numeric columns
    num_df = df.select_dtypes(include=[np.number])

    # To prevent memory overload, sample 100 columns
    if num_df.shape[1] > 100:
        print("[INFO] Too many columns — sampling 100 numeric columns for heatmap.")
        num_df = num_df.sample(n=100, axis=1)

    corr = num_df.corr().clip(-1, 1)

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Sampled Columns)")
    plt.show()


# ------------------------------------------
# MAIN
# ------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True,
                        help="Path to parquet dataset file.")
    args = parser.parse_args()

    df = load_dataset(args.file)
    df = ensure_date(df)

    inspect_basic(df)
    inspect_column_groups(df)
    inspect_missing_values(df)
    check_date_continuity(df)
    inspect_correlations(df)


if __name__ == "__main__":
    main()



