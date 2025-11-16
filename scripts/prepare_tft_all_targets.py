#!/usr/bin/env python3
"""
prepare_tft_all_targets.py

Creates TFT-ready train/val/test datasets for each company.
Target column = 'close' (company close price).
"""

import os
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import TorchNormalizer
import warnings

warnings.filterwarnings("ignore")

INPUT_DIR = "data/sanitised_final/company"
OUTPUT_DIR = "data/tft_ready"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def load_company_list():
    import pandas as pd
    df = pd.read_excel("data/australian_companies.xlsx")
    tickers = df["TICKER"].str.replace(".", "_", regex=False).tolist()
    return tickers

# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------
def main():
    tickers = load_company_list()

    print("\n===== COMPANY TICKERS =====")
    for t in tickers:
        print(" -", t)
    print("="*40)

    for t in tickers:
        print(f"\n================= {t} =================")

        file_path = f"{INPUT_DIR}/{t}_clean_sanitized.parquet"
        print("[INFO] Loading dataset:", file_path)

        if not os.path.exists(file_path):
            print(f"  [ERROR] File not found, skipping: {file_path}")
            continue

        df = pd.read_parquet(file_path)

        # Ensure Date column exists and sorted
        if "Date" not in df.columns:
            print(f"  [ERROR] No Date column in {t}, skipping.")
            continue

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # Add time index required by TFT
        df["time_idx"] = range(len(df))

        # Target detection
        TARGET = "close"   # <--- FIXED

        if TARGET not in df.columns:
            print(f"  [ERROR] Target column 'close' not found in {t}.")
            print("  Available columns:", df.columns[-20:])
            continue

        # Group ID (all companies are single-series)
        df["series"] = t

        # Train/val/test split
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.82)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        # All features except identifiers
        ignore_cols = ["Date", "series", "time_idx"]
        if TARGET in ignore_cols:
            ignore_cols.remove(TARGET)

        feature_cols = [c for c in df.columns if c not in ignore_cols]

        # Setup TFT dataset
        max_encoder_length = 60
        max_prediction_length = 5

        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=TARGET,
            group_ids=["series"],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,
            max_prediction_length=max_prediction_length,
            time_varying_unknown_reals=feature_cols,
            time_varying_known_reals=["time_idx"],
            target_normalizer=GroupNormalizer(groups=["series"]),
        )

        # Save output
        out_dir = f"{OUTPUT_DIR}/{t}"
        os.makedirs(out_dir, exist_ok=True)

        train_df.to_parquet(f"{out_dir}/train.parquet")
        val_df.to_parquet(f"{out_dir}/val.parquet")
        test_df.to_parquet(f"{out_dir}/test.parquet")

        print(f"  [OK] Saved train/val/test for {t} → {out_dir}/")

    print("\n[DONE] TFT dataset preparation complete.")


if __name__ == "__main__":
    main()
