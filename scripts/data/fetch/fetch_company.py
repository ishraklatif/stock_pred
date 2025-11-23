#!/usr/bin/env python3
"""
fetch_company.py

Fetches company OHLCV data using yfinance.
Saves files using safe canonical names (AGL_AX.parquet).
"""

import os
from datetime import datetime
import yaml
import yfinance as yf
import pandas as pd

from scripts.data.fetch.canonical_map import safe_filename

CONFIG_PATH = "config/data.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_yf(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # Flatten multi-index columns
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            c = c[0]
        cols.append(str(c).lower().replace(" ", "_"))

    df.columns = cols

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")

    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    return df[[c for c in keep if c in df.columns]]


def fetch_one(symbol, start, end):
    print(f"[INFO] Fetching {symbol} ...")
    df = yf.download(symbol, start=start, end=end, progress=False)

    if df.empty:
        print(f"[WARN] No data: {symbol}")
        return None

    return clean_yf(df)


def main():
    cfg = load_config()
    companies = cfg["companies"]["tickers_with_sectors"]

    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or datetime.today().strftime("%Y-%m-%d")

    out_dir = cfg["data"]["sources"]["companies_folder"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Saving to → {out_dir}")

    for symbol in companies.keys():
        df = fetch_one(symbol, start, end)
        if df is None or df.empty:
            continue

        safe = safe_filename(symbol)
        out = os.path.join(out_dir, f"{safe}.parquet")

        df.to_parquet(out, index=False)
        print(f"[OK] Saved {out}")

    print("\n[COMPLETE] Company fetch finished.\n")


if __name__ == "__main__":
    main()
