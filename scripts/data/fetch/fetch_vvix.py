#!/usr/bin/env python3
"""
fetch_vvix.py

Fetch VVIX index (volatility of VIX) from Yahoo Finance.

Yahoo ticker:
    ^VVIX

Output:
    data/raw_macro_extra/VVIX.parquet

Schema:
    date, open, high, low, close, adj_close, volume
"""

import os
from datetime import datetime

import yaml
import yfinance as yf
import pandas as pd

CONFIG_PATH = "config/data.yaml"
DEFAULT_OUT_DIR = "data/raw_macro_extra"


# ----------------------------------------------------------
# Config helpers
# ----------------------------------------------------------
def load_config(path: str = CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        print(f"[WARN] Config file not found at {path}, using defaults.")
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_data_dates(cfg: dict) -> tuple[str, str]:
    data_cfg = cfg.get("data", {})
    start = data_cfg.get("start_date", "2000-01-01")
    end = data_cfg.get("end_date") or datetime.today().strftime("%Y-%m-%d")
    return start, end


def get_out_dir(cfg: dict) -> str:
    sources = cfg.get("data", {}).get("sources", {})
    return sources.get("macro_extra_folder", DEFAULT_OUT_DIR)


# ----------------------------------------------------------
# Cleaning
# ----------------------------------------------------------
def clean_yf(df: pd.DataFrame) -> pd.DataFrame:
    """Standard OHLCV cleaner with forward-fill."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    flat_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            c = c[0]
        flat_cols.append(str(c).lower().replace(" ", "_"))
    df.columns = flat_cols

    if "date" not in df.columns:
        # Some yfinance versions use 'index' instead
        if df.index.name is not None:
            df = df.rename_axis("date").reset_index()
        else:
            raise ValueError("Could not find date column in Yahoo data")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in keep if c in df.columns]]

    # Forward-fill numeric columns
    num_cols = [c for c in df.columns if c != "date"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[num_cols] = df[num_cols].ffill()

    return df


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    cfg = load_config()
    start, end = get_data_dates(cfg)
    out_dir = get_out_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)

    ticker = "^VVIX"
    canonical = "VVIX"

    print(f"[INFO] Fetching VVIX from Yahoo Finance ({ticker}) ...")
    try:
        raw = yf.download(ticker, start=start, end=end, progress=False)
    except Exception as e:
        print(f"[ERROR] yfinance error for {ticker}: {e}")
        return

    if raw is None or raw.empty:
        print(f"[WARN] No data returned for {ticker}")
        return

    df = clean_yf(raw)
    if df.empty:
        print(f"[WARN] VVIX cleaned DataFrame is empty")
        return

    out_path = os.path.join(out_dir, f"{canonical}.parquet")
    df.to_parquet(out_path, index=False)

    print(f"[OK] Saved VVIX → {out_path} (rows={len(df)})")
    print("\n[COMPLETE] VVIX fetch finished.\n")


if __name__ == "__main__":
    main()
