#!/usr/bin/env python3
"""
prepare_multiseries_tft_dataset.py

Loads per-company parquet files from data/tft_ready,
sanitizes column names, fills missing values, attaches sector_id,
builds a multiseries dataset for TFT training,
adds time_idx, splits train/val/test, and saves parquet outputs.

Author: stock_pred
"""

import os
import pandas as pd
import numpy as np
import yaml

# =====================================================================
# CONFIG
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
    "dayofweek", "weekday", "weekofyear", "month",
    "is_month_start", "is_month_end",
    "is_quarter_end",
    "holiday_au", "holiday_us", "holiday_cn",
]


# =====================================================================
# SANITIZATION
# =====================================================================

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make all column names TFT-safe:
        - no dots `.`
        - no hyphens `-`
        - no equals `=`
    """
    df.columns = (
        df.columns
        .str.replace(".", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("=", "_", regex=False)
    )
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace all NaN and +/-inf with 0.
    TFT does NOT allow NaN in ANY real-valued feature.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    return df


# =====================================================================
# SECTOR MAP
# =====================================================================

def load_sector_map(yaml_path="config/data.yaml"):
    if not os.path.exists(yaml_path):
        print(f"[WARN] data.yaml not found at {yaml_path}")
        return {}

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    try:
        sector_map = cfg["companies"]["tickers_with_sectors"]
        print(f"[INFO] Loaded {len(sector_map)} sector mappings from data.yaml")
        return sector_map
    except Exception as e:
        print(f"[WARN] Failed to read sector mapping from data.yaml: {e}")
        return {}


# =====================================================================
# LOAD ALL COMPANIES
# =====================================================================

def load_all_companies():
    sector_map = load_sector_map()
    dfs = []
    missing_sectors = set()

    files = sorted(os.listdir(INPUT_DIR))

    for file in files:
        if not file.endswith(".parquet"):
            continue

        ticker = file.replace(".parquet", "")
        path = os.path.join(INPUT_DIR, file)

        print(f"[LOAD] {ticker} from {path}")
        df = pd.read_parquet(path)

        # 1 — Sanitize column names
        df = sanitize_columns(df)

        # 2 — Ensure Date exists
        if DATE_COL not in df.columns:
            raise KeyError(f"{DATE_COL} missing in file: {file}")

        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

        # 3 — Add series id
        df["series"] = ticker

        # 4 — Apply sector mapping
        if "sector_id" in df.columns:
            df["sector_id"] = df["sector_id"].astype(str)
        else:
            sector = sector_map.get(ticker, "Unknown")
            df["sector_id"] = sector
            if sector == "Unknown":
                missing_sectors.add(ticker)

        dfs.append(df)

    if missing_sectors:
        print("\n[WARN] These tickers had no sector in YAML → sector_id='Unknown':")
        for t in sorted(missing_sectors):
            print(" -", t)
        print()

    if not dfs:
        raise RuntimeError("No parquet files found in INPUT_DIR")

    df = pd.concat(dfs, ignore_index=True)

    # 5 — Replace ALL missing/inf → 0
    df = fill_missing(df)

    return df


# =====================================================================
# ADD TIME INDEX
# =====================================================================

def add_time_idx(df):
    df = df.sort_values(["series", DATE_COL]).reset_index(drop=True)
    df["time_idx"] = (
        df.groupby("series")
        .cumcount()
        .astype("int64")
    )
    return df


# =====================================================================
# INFER OBSERVED HISTORICAL FEATURES
# =====================================================================

def infer_observed_features(df):
    ignore = set(STATIC_CATEGORICALS + KNOWN_FUTURE + [DATE_COL, TARGET, "time_idx"])

    observed = [
        c for c in df.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(df[c])
    ]

    print(f"[INFO] Observed historical features detected = {len(observed)}")
    return observed


# =====================================================================
# GLOBAL SPLIT
# =====================================================================

def split(df):
    train = df[df[DATE_COL] <= TRAIN_END].copy()
    val   = df[(df[DATE_COL] >= VAL_START) & (df[DATE_COL] <= VAL_END)].copy()
    test  = df[df[DATE_COL] >= TEST_START].copy()

    if train.empty or val.empty or test.empty:
        raise RuntimeError("Train/val/test contains an empty split.")

    print(f"[SPLIT] Train {train[DATE_COL].min().date()} → {train[DATE_COL].max().date()}  ({len(train)})")
    print(f"[SPLIT] Val   {val[DATE_COL].min().date()} → {val[DATE_COL].max().date()}  ({len(val)})")
    print(f"[SPLIT] Test  {test[DATE_COL].min().date()} → {test[DATE_COL].max().date()}  ({len(test)})")

    return train, val, test


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("\n==============================================")
    print("   BUILDING MULTISERIES DATASET FOR TFT")
    print("==============================================\n")

    # Step 1 — Load & sanitize & fill
    df = load_all_companies()

    # Step 2 — Add time_idx
    df = add_time_idx(df)

    # Step 3 — Observed feature detection (debug only)
    observed = infer_observed_features(df)

    # Step 4 — Train/Val/Test split
    train, val, test = split(df)

    # Step 5 — Fill missing in each split
    train = fill_missing(train)
    val   = fill_missing(val)
    test  = fill_missing(test)

    # Step 6 — Save parquet outputs
    train_path = os.path.join(OUTPUT_DIR, "train.parquet")
    val_path   = os.path.join(OUTPUT_DIR, "val.parquet")
    test_path  = os.path.join(OUTPUT_DIR, "test.parquet")

    train.to_parquet(train_path)
    val.to_parquet(val_path)
    test.to_parquet(test_path)

    print("\n[OK] Saved:")
    print(f" - {train_path}")
    print(f" - {val_path}")
    print(f" - {test_path}")

    print("\nTFT Feature Groups:")
    print(" static_categorical_features  =", STATIC_CATEGORICALS)
    print(" known_future_features        =", KNOWN_FUTURE)
    print(" observed_historical_features =", len(observed))

    print("\n==============================================")
    print("          MULTISERIES DATA READY")
    print("==============================================\n")


if __name__ == "__main__":
    main()
