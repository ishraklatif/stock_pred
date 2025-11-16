#!/usr/bin/env python3
"""
fetch_data.py

Fetch historical market data using configuration-driven design.
Supports:
- Markets (ASX, SPY, FTSE, N225, China, etc.)
- Commodities (gold, oil, iron ore)
- FX pairs (AUDUSD, AUDJPY, AUDCNY)
"""

import os
import yaml
import yfinance as yf
from datetime import datetime
import pandas as pd

CONFIG_PATH = "config/config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def flatten_dict_lists(d):
    """Flattens nested dict of lists -> one flat list."""
    combined = []
    for v in d.values():
        if isinstance(v, list):
            combined.extend(v)
    return combined


def gather_all_symbols(cfg):
    symbols = []

    if cfg["include"]["markets"]:
        symbols.extend(flatten_dict_lists(cfg["markets"]))

    if cfg["include"]["commodities"]:
        symbols.extend(cfg["commodities"])

    if cfg["include"]["fx"]:
        symbols.extend(cfg["fx"])

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for s in symbols:
        if s not in seen:
            unique.append(s)
            seen.add(s)

    return unique


def fetch_single_symbol(symbol, start, end):
    """Download historical data for one symbol."""
    try:
        print(f"[INFO] Fetching: {symbol}")
        df = yf.download(symbol, start=start, end=end)

        if df.empty:
            print(f"[WARN] No data for {symbol}")
            return None

        df["symbol"] = symbol
        return df

    except Exception as e:
        print(f"[ERROR] Failed for {symbol}: {e}")
        return None


def save_output(df, symbol, cfg):
    out_folder = cfg["data"]["output_folder"]
    os.makedirs(out_folder, exist_ok=True)

    if cfg["options"].get("save_parquet", True):
        df.to_parquet(f"{out_folder}/{symbol}.parquet")

    if cfg["options"].get("save_csv", False):
        df.to_csv(f"{out_folder}/{symbol}.csv")

    print(f"[OK] Saved: {symbol}")


def main():
    cfg = load_config()
    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"]["end_date"] or datetime.today().strftime("%Y-%m-%d")

    symbols = gather_all_symbols(cfg)
    print(f"[INFO] Total symbols to fetch: {len(symbols)}")
    print("[INFO] Symbols:", symbols)

    for symbol in symbols:
        df = fetch_single_symbol(symbol, start_date, end_date)
        if df is not None:
            save_output(df, symbol, cfg)

    print("\n[COMPLETE] All data fetched successfully.")


if __name__ == "__main__":
    main()

