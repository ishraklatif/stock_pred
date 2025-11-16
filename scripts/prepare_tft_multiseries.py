#!/usr/bin/env python3

import os
import pandas as pd

INPUT_DIR = "data/tft_readyv2"     # your per-company folders
OUTPUT_DIR = "data/tft_ready_multiseries"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    all_train = []
    all_val = []
    all_test = []

    tickers = sorted(os.listdir(INPUT_DIR))

    for t in tickers:
        folder = os.path.join(INPUT_DIR, t)
        if not os.path.isdir(folder):
            continue

        print("Merging:", t)

        train_p = os.path.join(folder, "train.parquet")
        val_p = os.path.join(folder, "val.parquet")
        test_p = os.path.join(folder, "test.parquet")

        if not (os.path.exists(train_p) and os.path.exists(val_p) and os.path.exists(test_p)):
            print(f"  Skipping {t} (missing files)")
            continue

        # Load each file and keep the series label
        train = pd.read_parquet(train_p)
        val = pd.read_parquet(val_p)
        test = pd.read_parquet(test_p)

        train["series"] = t
        val["series"] = t
        test["series"] = t

        all_train.append(train)
        all_val.append(val)
        all_test.append(test)

    # Stack all
    train_df = pd.concat(all_train, ignore_index=True)
    val_df = pd.concat(all_val, ignore_index=True)
    test_df = pd.concat(all_test, ignore_index=True)

    # Save merged
    train_df.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"))
    val_df.to_parquet(os.path.join(OUTPUT_DIR, "val.parquet"))
    test_df.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"))

    print("\n[DONE] Multi-series TFT dataset created!")
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
