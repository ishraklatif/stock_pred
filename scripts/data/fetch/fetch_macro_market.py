#!/usr/bin/env python3
"""
fetch_macro_market.py

Fetches macro-market instruments such as:
VIX, DXY, commodities, FX pairs.
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
    print(f"[INFO] Fetching macro-market: {symbol}")
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        print(f"[WARN] No data for {symbol}")
        return None
    return clean_yf(df)


def main():
    cfg = load_config()

    tickers = cfg["market"]["tickers"]
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or datetime.today().strftime("%Y-%m-%d")

    out_dir = cfg["data"]["sources"]["market_folder"]
    os.makedirs(out_dir, exist_ok=True)

    for symbol in tickers:
        df = fetch_symbol(symbol, start, end)
        if df is None or df.empty:
            continue
        safe = symbol.replace("=", "_").replace("^", "")
        df.to_parquet(f"{out_dir}/{safe}.parquet", index=False)
        print(f"[OK] Saved {safe}.parquet")

    print("\n[COMPLETE] Macro-market fetch done.\n")


if __name__ == "__main__":
    main()
