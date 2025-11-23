#!/usr/bin/env python3

import sys
import os
import pandas as pd
from pathlib import Path

# Add repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT)

from scripts.data.merge.merge_all_features import (
    merge_single_ticker,
    load_config,
)


# -----------------------------------------------------------
# Helper loaders (mirror merge_all_features.main)
# -----------------------------------------------------------

def load_macro_block(cfg):
    macro_dir = cfg["data"]["processed"]["macro_folder"]
    frames = []

    for f in os.listdir(macro_dir):
        if f.endswith(".parquet"):
            name = Path(f).stem  # AXJO, GSPC, etc.
            df = pd.read_parquet(os.path.join(macro_dir, f))

            # Keep Date unprefixed
            df = df.rename(columns={
                c: f"{name}_{c}" for c in df.columns if c != "Date"
            })

            frames.append(df)

    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="Date", how="outer")

    return out



def load_market_block(cfg):
    market_dir = cfg["data"]["processed"]["market_folder"]
    frames = []

    for f in os.listdir(market_dir):
        if f.endswith(".parquet"):
            name = Path(f).stem
            df = pd.read_parquet(os.path.join(market_dir, f))

            df = df.rename(columns={
                c: f"{name}_{c}" for c in df.columns if c != "Date"
            })

            frames.append(df)

    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="Date", how="outer")

    return out



def load_calendar_block(cfg):
    cal_dir = cfg["data"]["processed"]["calendar_folder"]
    cal_path = os.path.join(cal_dir, "calendar_master.parquet")
    return pd.read_parquet(cal_path)


def load_sentiment_block(cfg):
    sent_dir = cfg["data"]["processed"]["news_sentiment_folder"]
    frames = []

    for f in os.listdir(sent_dir):
        if f.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(sent_dir, f))
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="Date", how="outer")

    return out


# -----------------------------------------------------------
# Main debug driver
# -----------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_single_merge.py <TICKER>")
        return

    ticker = sys.argv[1]
    print(f"[DEBUG] Testing merge for {ticker}")

    cfg = load_config()

    # Load everything
    macro_all = load_macro_block(cfg)
    market_all = load_market_block(cfg)
    calendar_df = load_calendar_block(cfg)
    sentiment_all = load_sentiment_block(cfg)

    companies_dir = cfg["data"]["processed"]["companies_folder"]
    out_dir = cfg["data"]["processed"]["tft_ready_folder"]

    # Look up sector
    sector_name = cfg["companies"]["tickers_with_sectors"].get(ticker, "Unknown")

    # Call merge function with correct argument order
    df = merge_single_ticker(
        ticker,
        sector_name,
        companies_dir,
        macro_all,
        market_all,
        calendar_df,
        sentiment_all,
        out_dir
    )

    if df is None or df.empty:
        print("[ERROR] Merge returned None or empty dataframe!")
        return

    print("\n[INFO] First 10 columns:", df.columns[:10].tolist())
    print("[INFO] Total columns:", len(df.columns))
    print("[INFO] DataFrame shape:", df.shape)

    # Validation
    if "series" not in df.columns:
        print("[ERROR] Missing 'series' column!")
    if "Date" not in df.columns:
        print("[ERROR] Missing 'Date' column!")

    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()
