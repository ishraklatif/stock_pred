#!/usr/bin/env python3
"""
fetch_sector.py

Fetch sector ETF prices (XLF, XLK, XLI, etc.) via yfinance.

Uses canonical_map.py to unify variants and save clean parquet files.

Output folder:
    data/raw_sector/
"""

import os
import yaml
import yfinance as yf
import pandas as pd

from scripts.data.fetch.canonical_map import canonical_name, safe_filename

CONFIG_PATH = "config/data.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_yf(df):
    """Standard OHLCV cleaner used across all fetch scripts."""
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
    df = df.sort_values("date")

    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    return df[[c for c in keep if c in df.columns]]


def fetch_one(symbol, start, end):
    print(f"[INFO] Fetching sector ETF: {symbol}")
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        print(f"[WARN] No data for {symbol}")
        return None
    return clean_yf(df)


def main():
    cfg = load_config()

    tickers = cfg["sector"]["tickers"]
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"]

    out_dir = cfg["data"]["sources"]["sector_folder"]
    os.makedirs(out_dir, exist_ok=True)

    for sym in tickers:
        Canon = canonical_name(sym)
        df = fetch_one(sym, start, end)
        if df is None or df.empty:
            continue

        fname = f"{safe_filename(Canon)}.parquet"
        out = os.path.join(out_dir, fname)

        df.to_parquet(out, index=False)
        print(f"[OK] Saved {out}")

    print("\n[COMPLETE] Sector ETF fetch finished.\n")


if __name__ == "__main__":
    main()

#python3 -m scripts.data.fetch.fetch_sector

