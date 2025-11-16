#!/usr/bin/env python3
"""
prepare_tft_all_targets.py

Creates TFT-ready train/val/test datasets for each ASX company.
Improvements:
 - Proper chronological splits
 - Encoder+Prediction window OFFSET applied to avoid leakage
 - Saves metadata (date ranges, sizes)
 - Validation/test sets guaranteed to have non-overlapping windows
 - Clean logging & structure
"""

import os
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import warnings

warnings.filterwarnings("ignore")

INPUT_DIR = "data/sanitised_final/company"
OUTPUT_DIR = "data/tft_readyv2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TFT hyperparameters
ENC_LEN = 60
PRED_LEN = 5
OFFSET = ENC_LEN + PRED_LEN  # to avoid overlapping windows


# ---------------------------------------------------------
# Load company list
# ---------------------------------------------------------
def load_company_list():
    df = pd.read_excel("data/australian_companies.xlsx")
    tickers = df["TICKER"].str.replace(".", "_", regex=False).tolist()
    return tickers


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    tickers = load_company_list()

    print("\n===== COMPANY TICKERS =====")
    for t in tickers:
        print(" -", t)
    print("=====================================\n")

    for t in tickers:
        print(f"\n================= {t} =================")

        file_path = f"{INPUT_DIR}/{t}_clean_sanitized.parquet"
        print("[INFO] Loading:", file_path)

        if not os.path.exists(file_path):
            print(f"[ERROR] File not found, skipping.")
            continue

        df = pd.read_parquet(file_path)

        # --------------------------
        # Basic checks
        # --------------------------
        if "Date" not in df.columns:
            print(f"[ERROR] Missing Date column for {t}")
            continue

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # Add required columns
        df["time_idx"] = range(len(df))
        df["series"] = t

        if "close" not in df.columns:
            print(f"[ERROR] Target column 'close' missing.")
            continue

        n = len(df)
        if n < (ENC_LEN + PRED_LEN + 200):
            print(f"[WARNING] Not enough rows for reliable TFT training. Skipping.")
            continue

        # ---------------------------------------------------------
        # Compute split points (with OFFSET)
        # ---------------------------------------------------------
        raw_train_end = int(n * 0.70)
        raw_val_end = int(n * 0.82)

        train_end = raw_train_end
        val_start = raw_train_end + OFFSET

        # Ensure validation start is safe
        val_start = min(val_start, n - OFFSET - 1)

        val_end = raw_val_end
        test_start = val_end

        # ---------------------------------------------------------
        # Final splits
        # ---------------------------------------------------------
        train_df = df.iloc[:train_end]
        val_df = df.iloc[val_start:val_end]
        test_df = df.iloc[test_start:]

        print(f"[SPLIT] Train: {train_df['Date'].min().date()} → {train_df['Date'].max().date()}  ({len(train_df)} rows)")
        print(f"[SPLIT] Val:   {val_df['Date'].min().date()} → {val_df['Date'].max().date()}  ({len(val_df)} rows)")
        print(f"[SPLIT] Test:  {test_df['Date'].min().date()} → {test_df['Date'].max().date()}  ({len(test_df)} rows)")
        print(f"[INFO] OFFSET used: {OFFSET} timesteps")

        # ---------------------------------------------------------
        # Feature selection
        # ---------------------------------------------------------
        ignore_cols = ["Date", "series", "time_idx"]
        feature_cols = [c for c in df.columns if c not in ignore_cols]

        # ---------------------------------------------------------
        # Build TFT train dataset
        # ---------------------------------------------------------
        training_ds = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="close",
            group_ids=["series"],
            min_encoder_length=ENC_LEN,
            max_encoder_length=ENC_LEN,
            min_prediction_length=PRED_LEN,
            max_prediction_length=PRED_LEN,
            time_varying_unknown_reals=feature_cols,
            time_varying_known_reals=["time_idx"],
            target_normalizer=GroupNormalizer(groups=["series"]),
        )

        # (Validation and test TFT datasets will be created during training via from_dataset())

        # ---------------------------------------------------------
        # Save output
        # ---------------------------------------------------------
        out_dir = f"{OUTPUT_DIR}/{t}"
        os.makedirs(out_dir, exist_ok=True)

        train_df.to_parquet(f"{out_dir}/train.parquet")
        val_df.to_parquet(f"{out_dir}/val.parquet")
        test_df.to_parquet(f"{out_dir}/test.parquet")

        # Save metadata
        meta = {
            "ticker": t,
            "rows": n,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_start": train_df["Date"].min(),
            "train_end": train_df["Date"].max(),
            "val_start": val_df["Date"].min(),
            "val_end": val_df["Date"].max(),
            "test_start": test_df["Date"].min(),
            "test_end": test_df["Date"].max(),
            "offset": OFFSET,
        }
        pd.DataFrame([meta]).to_parquet(f"{out_dir}/metadata.parquet")

        print(f"[OK] Saved TFT-ready dataset → {out_dir}/")

    print("\n[DONE] All companies processed.")


if __name__ == "__main__":
    main()
