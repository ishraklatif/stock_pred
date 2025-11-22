#!/usr/bin/env python3
"""
fetch_company.py

Fetches historical OHLCV for all ASX company tickers defined in config/data.yaml.
Uses canonical yfinance cleaning logic.
"""

import os
from datetime import datetime
import yaml
import yfinance as yf
import pandas as pd

CONFIG_PATH = "config/data.yaml"


# ---------------------------------------------------------
# Load config
# ---------------------------------------------------------
def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------
# Canonical OHLCV cleaner
# ---------------------------------------------------------
def clean_yf(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    # flatten possible MultiIndex
    flat = []
    for col in df.columns:
        if isinstance(col, tuple):
            col = col[0]
        flat.append(str(col).lower().replace(" ", "_"))
    df.columns = flat

    if "date" not in df.columns:
        raise RuntimeError("No 'date' column found after flattening yfinance frame.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in keep if c in df.columns]]

    return df


# ---------------------------------------------------------
# Fetch a company
# ---------------------------------------------------------
def fetch_company(ticker, start, end):
    print(f"[INFO] Fetching {ticker} ...")
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False
        )
    except Exception as e:
        print(f"[ERROR] yfinance error for {ticker}: {e}")
        return None

    if df is None or df.empty:
        print(f"[WARN] No data for {ticker}")
        return None

    df = clean_yf(df)
    return df


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    cfg = load_config()

    tickers = list(cfg["companies"]["tickers_with_sectors"].keys())
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or datetime.today().strftime("%Y-%m-%d")

    out_dir = cfg["data"]["sources"]["companies_folder"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Saving to → {out_dir}")

    for ticker in tickers:
        df = fetch_company(ticker, start, end)
        if df is None or df.empty:
            continue

        path = f"{out_dir}/{ticker}.parquet"
        df.to_parquet(path, index=False)
        print(f"[OK] Saved {path}")

    print("\n[COMPLETE] Company fetch finished.\n")


if __name__ == "__main__":
    main()
