#!/usr/bin/env python3
"""
fetch_move_index.py

Correct MOVE Index fetcher using Yahoo Finance (^MOVE).

MOVE cannot be fetched from FRED — it is NOT hosted there.
This script matches the VVIX fetcher style.

Output:
    data/raw_macro_extra/MOVE.parquet

Schema:
    date, open, high, low, close, adj_close, volume
"""

import os
import yaml
import yfinance as yf
import pandas as pd
from pathlib import Path

# ----------------------------------------------------------
# Load config
# ----------------------------------------------------------
CONFIG_PATH = "config/data.yaml"

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------
# Clean Yahoo OHLCV dataframe
# ----------------------------------------------------------
def clean_yf(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    flat_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            c = c[0]
        flat_cols.append(str(c).lower().replace(" ", "_"))

    df.columns = flat_cols

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")

    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    return df[[c for c in keep if c in df.columns]]


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    print("\n[INFO] Fetching MOVE Index from Yahoo Finance (^MOVE) ...")

    cfg = load_config()
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or None

    ticker = "^MOVE"

    try:
        raw = yf.download(ticker, start=start, end=end, progress=False)
    except Exception as e:
        print(f"[ERROR] yfinance error for ^MOVE: {e}")
        return

    if raw is None or raw.empty:
        print("[WARN] No data for MOVE (^MOVE)")
        return

    df = clean_yf(raw)

    out_dir = "data/raw_macro_extra"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "MOVE.parquet")
    df.to_parquet(out_path, index=False)

    print(f"[OK] Saved MOVE → {out_path} (rows={len(df)})")
    print("\n[COMPLETE] MOVE fetch finished.\n")


if __name__ == "__main__":
    main()
