#!/usr/bin/env python3
"""
Automated Feature Reduction Pipeline for TFT
--------------------------------------------

This script reduces your 1200+ feature dataset into a compact,
high-signal feature matrix suitable for Temporal Fusion Transformer.

Steps:
 1. Remove duplicated merge artifacts (_x, _y)
 2. Reduce indicator explosion for each macro ticker
 3. Reduce indicator explosion for company OHLCV
 4. Reduce sentiment features
 5. Drop sparse features (>40% NaN)
 6. Drop zero/near-zero variance features
 7. Drop highly correlated features (|corr| > 0.95)
 8. Save final reduced dataset + report

Output:
   data/tft_reduced/train.parquet
   data/tft_reduced/val.parquet
   data/tft_reduced/test.parquet
   data/tft_reduced/feature_reduction_report.txt
"""

import os
import pandas as pd
import numpy as np

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------

INPUT_DIR = "data/tft_ready_multiseries"
OUTPUT_DIR = "data/tft_reduced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Selected macro indicators to retain per ticker (10 total)
MACRO_KEEP = [
    "close", "log_return",
    "sma_20", "sma_50",
    "ema_20",
    "rsi_14",
    "macd", "macd_signal",
    "vol_20",
    "obv"
]

# Company-level indicators to retain (approx 25)
COMPANY_KEEP = [
    "close", "open", "high", "low", "volume",
    "log_return",
    "sma_20", "sma_50", "sma_100",
    "ema_20", "ema_50",
    "rsi_14",
    "macd", "macd_signal",
    "vol_20",
    "obv",
    "ret_1", "ret_5", "ret_10"
]

# Sentiment features to retain
SENTIMENT_KEEP = ["sent_mean", "sent_vol"]

# Correlation threshold
CORR_THRESHOLD = 0.95


# --------------------------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------------------------

def load_datasets():
    train = pd.read_parquet(f"{INPUT_DIR}/train.parquet")
    val   = pd.read_parquet(f"{INPUT_DIR}/val.parquet")
    test  = pd.read_parquet(f"{INPUT_DIR}/test.parquet")
    return train, val, test


# --------------------------------------------------------------------
# STEP 1 — Remove duplicated _x / _y merge artifacts
# --------------------------------------------------------------------

def fix_merge_duplicates(df, report):
    y_cols = [c for c in df.columns if c.endswith("_y")]
    x_cols = [c for c in df.columns if c.endswith("_x")]

    report.append(f"Dropping duplicated merge cols (_y): {len(y_cols)}")

    df = df.drop(columns=y_cols)

    rename_map = {c: c.replace("_x", "") for c in x_cols}
    df = df.rename(columns=rename_map)

    return df


# --------------------------------------------------------------------
# STEP 2 — Select useful indicators by pattern
# --------------------------------------------------------------------

def reduce_indicator_explosion(df, report):
    drop_cols = []

    for col in df.columns:
        # Skip category columns
        if col in ["series", "sector_id", "Date"]:
            continue

        # Determine ticker prefix (before first underscore)
        if "_" not in col:
            continue

        parts = col.split("_")
        ticker_prefix = parts[0]  # e.g. AXJO, AUDUSD, 000001_SS

        # Extract raw indicator name (suffix)
        ind = "_".join(parts[1:])  # e.g. ema_10, sma_5, bb_lower

        # Detect sentiment
        if ind.startswith(("pos", "neg", "neu", "min", "max")):
            drop_cols.append(col)
            continue
        if "sent_" in ind and not any(k in ind for k in SENTIMENT_KEEP):
            drop_cols.append(col)
            continue

        # Company features (no prefix or common ASX tickers)
        if ticker_prefix in ["close", "open", "high", "low", "volume"]:
            continue

        # Macro indicator reduction: keep only MACRO_KEEP
        if ticker_prefix.upper() in ["AXJO", "GSPC", "FTSE", "N225", "HSI",
                                     "DX", "DX-Y", "AUDUSD", "AUDJPY",
                                     "GC", "CL", "SI", "HG", "DBC", "VIX", 
                                     "000001", "000300"]:
            keep = False
            for k in MACRO_KEEP:
                if ind.startswith(k):
                    keep = True
                    break
            if not keep:
                drop_cols.append(col)
            continue

    drop_cols = sorted(list(set(drop_cols)))
    report.append(f"Dropping indicator explosion cols: {len(drop_cols)}")

    return df.drop(columns=drop_cols)


# --------------------------------------------------------------------
# STEP 3 — Drop sparse features (too many NaN)
# --------------------------------------------------------------------

def drop_sparse(df, report, threshold=0.40):
    nan_fraction = df.isna().mean()
    sparse_cols = nan_fraction[nan_fraction > threshold].index.tolist()

    report.append(f"Dropping sparse features (>40% NaN): {len(sparse_cols)}")

    return df.drop(columns=sparse_cols)


# --------------------------------------------------------------------
# STEP 4 — Drop zero-variance features
# --------------------------------------------------------------------

def drop_low_variance(df, report):
    std = df.std(numeric_only=True)
    zero_var_cols = std[std < 1e-9].index.tolist()

    report.append(f"Dropping low-variance features: {len(zero_var_cols)}")

    return df.drop(columns=zero_var_cols)


# --------------------------------------------------------------------
# STEP 5 — Drop correlated redundant features (>0.95)
# --------------------------------------------------------------------

def drop_correlated(df, report, threshold=CORR_THRESHOLD):
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    report.append(f"Dropping highly correlated features (>{threshold}): {len(to_drop)}")

    return df.drop(columns=to_drop)


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------

def main():
    train, val, test = load_datasets()
    report = []

    # STEP 1: Remove _x / _y
    train = fix_merge_duplicates(train, report)
    val   = fix_merge_duplicates(val, report)
    test  = fix_merge_duplicates(test, report)

    # STEP 2: Reduce indicator explosion
    train = reduce_indicator_explosion(train, report)
    val   = reduce_indicator_explosion(val, report)
    test  = reduce_indicator_explosion(test, report)

    # STEP 3: Drop sparse (>40% NaN)
    train = drop_sparse(train, report)
    val   = drop_sparse(val, report)
    test  = drop_sparse(test, report)

    # STEP 4: Drop low variance
    train = drop_low_variance(train, report)
    val   = drop_low_variance(val, report)
    test  = drop_low_variance(test, report)

    # STEP 5: Drop correlated (>0.95)
    train = drop_correlated(train, report)
    val   = drop_correlated(val, report)
    test  = drop_correlated(test, report)

    # Save processed dataset
    train.to_parquet(f"{OUTPUT_DIR}/train.parquet")
    val.to_parquet(f"{OUTPUT_DIR}/val.parquet")
    test.to_parquet(f"{OUTPUT_DIR}/test.parquet")

    # Save reduction report
    with open(f"{OUTPUT_DIR}/feature_reduction_report.txt", "w") as f:
        for line in report:
            f.write(line + "\n")

    print("\n[OK] Automated feature reduction complete.")
    print(f"Reduced dataset saved to: {OUTPUT_DIR}/")
    print("See feature_reduction_report.txt for details.\n")


if __name__ == "__main__":
    main()
