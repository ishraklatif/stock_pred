#!/usr/bin/env python3
"""
fetch_macro_main.py

Fetches global macro index OHLCV data:
(^AXJO, ^GSPC, ^FTSE, ^N225, etc.)
"""

import os
from datetime import datetime
import yaml
import yfinance as yf
import pandas as pd

CONFIG_PATH = "config/data.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_yf(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    flat = []
    for col in df.columns:
        if isinstance(col, tuple):
            col = col[0]
        flat.append(str(col).lower().replace(" ", "_"))
    df.columns = flat

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")

    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    return df[[c for c in keep if c in df.columns]]


def fetch_symbol(symbol, start, end):
    print(f"[INFO] Fetching macro: {symbol}")
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        print(f"[WARN] No data for {symbol}")
        return None
    return clean_yf(df)


def main():
    cfg = load_config()

    tickers = cfg["macro"]["tickers"]
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or datetime.today().strftime("%Y-%m-%d")

    out_dir = cfg["data"]["sources"]["macro_folder"]
    os.makedirs(out_dir, exist_ok=True)

    for symbol in tickers:
        df = fetch_symbol(symbol, start, end)
        if df is None or df.empty:
            continue
        out = f"{out_dir}/{symbol.replace('=', '_').replace('^', '')}.parquet"
        df.to_parquet(out, index=False)
        print(f"[OK] Saved → {out}")

    print("\n[COMPLETE] Macro fetch done.\n")


if __name__ == "__main__":
    main()
