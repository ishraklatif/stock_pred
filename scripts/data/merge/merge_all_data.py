#!/usr/bin/env python3
"""
merge_all_features.py

Produces final TFT-ready multiseries dataset for each company.

Merges:
- processed companies (indicators)
- processed macro (indicators)
- processed macro-market (indicators)
- news sentiment features
- global calendar_master

Output:
    data/tft_ready/<TICKER>.parquet

Config (config/data.yaml):

data:
  processed:
    companies_folder: "data/processed_companies"
    macro_folder: "data/processed_macro"
    market_folder: "data/processed_macro_market"
    calendar_folder: "data/processed_calendar"
    news_sentiment_folder: "data/news/sentiment"
    tft_ready_folder: "data/tft_ready"
"""

import os
from pathlib import Path

import yaml
import pandas as pd

CONFIG_PATH = "config/data.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_parquet_dict(directory: str):
    """
    Load all parquet files from a directory into a dict:
        {name_stem: DataFrame}
    """
    frames = {}
    root = Path(directory)
    if not root.exists():
        return frames

    for f in root.glob("*.parquet"):
        name = f.stem
        df = pd.read_parquet(f)
        frames[name] = df
    return frames


def ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'Date' column exists and is datetime, sorted.
    Accept 'Date' or 'date' or datetime index.
    """
    df = df.copy()

    if "Date" in df.columns:
        pass
    elif "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "Date"})
    else:
        raise RuntimeError("No Date/date column or datetime index found.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Prefix all non-Date columns with <prefix>_ to avoid collisions.
    """
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        if col == "Date":
            continue
        rename_map[col] = f"{prefix}_{col}"
    return df.rename(columns=rename_map)


def build_global_macro_frame(
    macro_dir: str,
    market_dir: str,
    cal_dir: str,
    news_dir: str,
) -> pd.DataFrame:
    """
    Build the global macro+market+news+calendar backbone.
    """
    print("\n[INFO] Building global macro backbone...")

    # --- load calendar master ---
    cal_path = Path(cal_dir) / "calendar_master.parquet"
    if not cal_path.exists():
        raise RuntimeError(f"calendar_master not found at {cal_path}")
    calendar_df = pd.read_parquet(cal_path)
    calendar_df = ensure_date(calendar_df)

    global_df = calendar_df.copy()

    # --- macro indicators ---
    macro_frames = load_parquet_dict(macro_dir)
    print(f"[INFO] Macro indicator frames: {list(macro_frames.keys())}")

    for symbol, df in macro_frames.items():
        df = ensure_date(df)
        df = prefix_columns(df, symbol)
        global_df = global_df.merge(df, on="Date", how="left")

    # --- macro-market indicators ---
    market_frames = load_parquet_dict(market_dir)
    print(f"[INFO] Macro-market frames: {list(market_frames.keys())}")

    for symbol, df in market_frames.items():
        df = ensure_date(df)
        df = prefix_columns(df, symbol)
        global_df = global_df.merge(df, on="Date", how="left")

    # --- news sentiment (already prefixed by asset name) ---
    news_frames = load_parquet_dict(news_dir)
    print(f"[INFO] News sentiment frames: {list(news_frames.keys())}")

    for asset, df in news_frames.items():
        df = ensure_date(df)
        global_df = global_df.merge(df, on="Date", how="left")

    # final sort
    global_df = global_df.sort_values("Date").reset_index(drop=True)

    print("[INFO] Global macro backbone shape:", global_df.shape)
    return global_df


def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    processed = data_cfg["processed"]

    comp_dir = processed["companies_folder"]
    macro_dir = processed["macro_folder"]
    market_dir = processed["market_folder"]
    cal_dir = processed["calendar_folder"]
    news_dir = processed["news_sentiment_folder"]
    out_dir = processed["tft_ready_folder"]

    os.makedirs(out_dir, exist_ok=True)

    # 1) Build global backbone once
    global_df = build_global_macro_frame(
        macro_dir=macro_dir,
        market_dir=market_dir,
        cal_dir=cal_dir,
        news_dir=news_dir,
    )

    # 2) Load processed company indicator frames
    companies = load_parquet_dict(comp_dir)
    print(f"\n[INFO] Companies loaded: {len(companies)}")

    for ticker, df in companies.items():
        print(f"[INFO] Merging TFT-ready dataset for: {ticker}")

        df = ensure_date(df)

        merged = df.merge(global_df, on="Date", how="left").sort_values("Date")
        merged = merged.reset_index(drop=True)

        out_path = os.path.join(out_dir, f"{ticker}.parquet")
        merged.to_parquet(out_path, index=False)
        print(f"[OK] Saved → {out_path}")

    print("\n[DONE] All TFT multiseries datasets generated.\n")


if __name__ == "__main__":
    main()
