#!/usr/bin/env python3
"""
fetch_companies.py

Reads company tickers from australian_companies.xlsx and fetches their historical OHLCV data.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

EXCEL_PATH = "data/australian_companies.xlsx"
OUTPUT_DIR = "data/raw_companies"
START = "2000-01-01"
END = datetime.today().strftime("%Y-%m-%d")


# ---------------------------------------------
# Load company tickers
# ---------------------------------------------

def load_tickers():
    df = pd.read_excel(EXCEL_PATH)

    # Accept both "TICKER" and "Ticker"
    if "TICKER" in df.columns:
        col = "TICKER"
    elif "Ticker" in df.columns:
        col = "Ticker"
    else:
        raise ValueError(
            f"❌ Could not find TICKER column in Excel. Columns: {df.columns.tolist()}"
        )

    tickers = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    print(f"[INFO] Loaded {len(tickers)} company tickers.")
    return tickers


# ---------------------------------------------
# Fetch one company's OHLCV data
# ---------------------------------------------

def fetch_company(ticker: str):
    print(f"[INFO] Fetching {ticker} ...")

    df = yf.download(
        ticker,
        start=START,
        end=END,
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        print(f"[WARN] No data for {ticker}")
        return None

    # Convert DatetimeIndex → date column
    df = df.reset_index()

    # Flatten possible MultiIndex columns
    flat = []
    for col in df.columns:
        if isinstance(col, tuple):
            flat.append(col[0].lower().replace(" ", "_"))
        else:
            flat.append(str(col).lower().replace(" ", "_"))
    df.columns = flat

    # Require 'date'
    if "date" not in df.columns:
        raise RuntimeError(f"❌ No 'date' found after flattening for {ticker}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # Keep standard OHLCV columns only
    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in keep if c in df.columns]]

    return df


# ---------------------------------------------
# Main
# ---------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tickers = load_tickers()

    for ticker in tickers:
        df = fetch_company(ticker)
        if df is not None:
            out = f"{OUTPUT_DIR}/{ticker}.parquet"
            df.to_parquet(out, index=False)
            print(f"[OK] Saved: {out}")

    print("[COMPLETE] Finished fetching all company data.")


if __name__ == "__main__":
    main()

# python3 scripts/fetch_company_data.py
