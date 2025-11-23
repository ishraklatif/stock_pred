#!/usr/bin/env python3
"""
prepare_multiseries_tft_dataset.py

Builds a clean, multiseries TFT-ready dataset from individual merged
parquet files inside data/tft_ready/.

Steps:
  1. Load per-company parquet files
  2. Sanitize column names
  3. Add sector_id (from config/data.yaml)
  4. Attach series column
  5. Add time_idx (per series)
  6. Drop constant/all-NaN numeric features
  7. Infer observed historical features
  8. Split into train/val/test
  9. Forward-fill inside each split (per series)
 10. Save train/val/test parquet files

Output directory:
    data/tft_ready_multiseries/
"""

import os
import pandas as pd
import numpy as np
import yaml

# =====================================================================
# PATHS
# =====================================================================

INPUT_DIR = "data/tft_ready"
OUTPUT_DIR = "data/tft_ready_multiseries"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE_COL = "Date"
TARGET = "close"

TRAIN_END  = pd.Timestamp("2020-03-18")
VAL_START  = pd.Timestamp("2020-06-18")
VAL_END    = pd.Timestamp("2022-06-24")
TEST_START = pd.Timestamp("2022-06-27")

STATIC_CATEGORICALS = ["series", "sector_id"]

KNOWN_FUTURE = [
    "day_of_week", "week_of_year", "month", "quarter", "day_of_month",
    "is_month_end", "is_quarter_end", "is_year_end",
    "is_aus_holiday", "is_us_holiday", "is_china_holiday",
    "dist_to_aus_holiday", "dist_to_us_holiday", "dist_to_china_holiday",
]


# =====================================================================
# COLUMN SANITIZATION
# =====================================================================

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove illegal characters:
        - replace anything not [0-9a-zA-Z_] with "_"
    Safest for TFT + parquet.
    """
    df.columns = (
        df.columns
        .str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
    )
    return df


# =====================================================================
# SECTOR MAPPING
# =====================================================================

def load_sector_map(yaml_path="config/data.yaml"):
    """Load sector mapping from YAML."""
    if not os.path.exists(yaml_path):
        print(f"[WARN] data.yaml not found at {yaml_path}")
        return {}

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    try:
        sector_map = cfg["companies"]["tickers_with_sectors"]
        print(f"[INFO] Loaded {len(sector_map)} sector mappings.")
        return sector_map
    except Exception as e:
        print(f"[WARN] Could not load sector mapping: {e}")
        return {}


# =====================================================================
# LOAD ALL COMPANIES
# =====================================================================

def load_all_companies():
    sector_map = load_sector_map()
    dfs = []
    missing = set()

    files = sorted(os.listdir(INPUT_DIR))

    if not files:
        raise RuntimeError("No files found in data/tft_ready/")

    for file in files:
        if not file.endswith(".parquet"):
            continue

        ticker = file.replace(".parquet", "").replace("_", ".")

        path = os.path.join(INPUT_DIR, file)

        print(f"[LOAD] {ticker} from {path}")
        df = pd.read_parquet(path)

        # Sanitize
        df = sanitize_columns(df)

        if DATE_COL not in df.columns:
            raise KeyError(f"Missing {DATE_COL} in {file}")

        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df["series"] = ticker

        # Attach sector_id
        sector = sector_map.get(ticker, "Unknown")
        df["sector_id"] = str(sector)
        if sector == "Unknown":
            missing.add(ticker)

        # Enforce categorical sector_id
        df["sector_id"] = df["sector_id"].astype("category")

        dfs.append(df)

    if missing:
        print("\n[WARN] Missing sector_id for:")
        for t in sorted(missing):
            print(" -", t)
        print("Assigned 'Unknown'.\n")

    final = pd.concat(dfs, ignore_index=True)
    return final


# =====================================================================
# ADD TIME INDEX
# =====================================================================

def add_time_idx(df: pd.DataFrame):
    df = df.sort_values(["series", DATE_COL]).reset_index(drop=True)
    df["time_idx"] = (
        df.groupby("series")
        .cumcount()
        .astype("int64")
    )
    return df


# =====================================================================
# DROP CONSTANT / ALL-NaN FEATURES
# =====================================================================

def drop_bad_features(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    zero_var = []
    all_nan = []

    for c in numeric_cols:
        if c == TARGET:
            continue

        col = df[c].replace([np.inf, -np.inf], np.nan)

        if col.isna().all():
            all_nan.append(c)
        elif col.nunique(dropna=True) <= 1:
            zero_var.append(c)

    to_drop = sorted(set(zero_var + all_nan))

    if to_drop:
        print(f"[CLEAN] Dropping {len(to_drop)} constant/all-NaN features:")
        for c in to_drop:
            print("  -", c)
        df = df.drop(columns=to_drop)

    return df, to_drop


# =====================================================================
# INFER OBSERVED HISTORICAL FEATURES
# =====================================================================

def infer_observed_features(df: pd.DataFrame):
    ignore = set(STATIC_CATEGORICALS + KNOWN_FUTURE + [DATE_COL, TARGET, "time_idx"])

    observed = [
        c for c in df.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(df[c])
    ]

    print(f"[INFO] Observed historical features detected = {len(observed)}")
    return observed


# =====================================================================
# GLOBAL TRAIN/VAL/TEST SPLIT
# =====================================================================

def split(df: pd.DataFrame):
    train = df[df[DATE_COL] <= TRAIN_END].copy()
    val   = df[(df[DATE_COL] >= VAL_START) & (df[DATE_COL] <= VAL_END)].copy()
    test  = df[df[DATE_COL] >= TEST_START].copy()

    if train.empty or val.empty or test.empty:
        raise RuntimeError("Empty split detected. Check your dates.")

    print(f"[SPLIT] Train: {len(train)} rows")
    print(f"[SPLIT] Val:   {len(val)} rows")
    print(f"[SPLIT] Test:  {len(test)} rows")

    return train, val, test


# =====================================================================
# PER-SPLIT MISSING VALUE HANDLING
# =====================================================================

def fill_split(split: pd.DataFrame):
    split = split.sort_values(["series", DATE_COL]).reset_index(drop=True)

    numeric_cols = split.select_dtypes(include=[np.number]).columns
    split[numeric_cols] = split[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Forward fill per series
    split[numeric_cols] = (
        split.groupby("series")[numeric_cols]
        .apply(lambda g: g.ffill())
        .reset_index(level=0, drop=True)
    )

    # Remaining NaN → 0.0
    split[numeric_cols] = split[numeric_cols].fillna(0.0)

    return split


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("\n==============================================")
    print("   BUILDING MULTISERIES DATASET FOR TFT")
    print("==============================================\n")

    df = load_all_companies()
    df = add_time_idx(df)
    df, dropped = drop_bad_features(df)
    observed = infer_observed_features(df)

    train, val, test = split(df)

    train = fill_split(train)
    val   = fill_split(val)
    test  = fill_split(test)

    train.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"))
    val.to_parquet(os.path.join(OUTPUT_DIR, "val.parquet"))
    test.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"))

    print("\n[OK] Saved TFT datasets:")
    print(f" - {OUTPUT_DIR}/train.parquet")
    print(f" - {OUTPUT_DIR}/val.parquet")
    print(f" - {OUTPUT_DIR}/test.parquet")

    print("\nTFT Feature Groups:")
    print(" static_categorical_features  =", STATIC_CATEGORICALS)
    print(" known_future_features        =", KNOWN_FUTURE)
    print(" observed_historical_features =", len(observed))

    print("\n==============================================")
    print("          MULTISERIES DATA READY")
    print("==============================================\n")


if __name__ == "__main__":
    main()
