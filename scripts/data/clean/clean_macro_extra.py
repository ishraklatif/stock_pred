#!/usr/bin/env python3
"""
clean_macro_extra.py

Cleans and normalizes all macro-extra series:
    MOVE, VVIX,
    HY_SPREAD, IG_SPREAD, TED_SPREAD,
    US3M, US5Y, US30Y

Rules:
- Convert to datetime
- Sort by date
- Resample to daily
- Forward-fill + back-fill
- Canonical naming
- Save to data/processed_macro_extra/

Output schema:
    date, value, series, region, source
or
    date, open, high, low, close, adj_close, volume, series
"""

import os
import pandas as pd
from pathlib import Path


RAW_DIR = "data/raw_macro_extra"
OUT_DIR = "data/processed_macro_extra"

SERIES_TYPES = {
    "MOVE": "yahoo_ohlcv",
    "VVIX": "yahoo_ohlcv",
    "HY_SPREAD": "fred_value",
    "IG_SPREAD": "fred_value",
    "TED_SPREAD": "fred_value",
    "US3M": "fred_value",
    "US5Y": "fred_value",
    "US30Y": "fred_value",
}


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_file(name: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{name}.parquet")
    if not os.path.exists(path):
        print(f"[WARN] Missing raw file: {path}")
        return None
    return pd.read_parquet(path)


def clean_value_series(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Clean series with schema:
        date, value, series, region, source
    """
    if df is None or df.empty:
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df = df.set_index("date").resample("1D").ffill().bfill().reset_index()

    df["series"] = name
    df["region"] = df.get("region", "US")
    df["source"] = df.get("source", "FRED")

    return df[["date", "value", "series", "region", "source"]]


def clean_ohlcv(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Clean OHLCV Yahoo-style series:
        date, open, high, low, close, [adj_close], [volume]
    Some tickers may not have adj_close.
    """
    if df is None or df.empty:
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # Daily resample with ffill/bfill to remove gaps
    df = df.set_index("date").resample("1D").ffill().bfill().reset_index()
    df["series"] = name

    # Only keep columns that actually exist
    cols_order = ["date", "open", "high", "low", "close", "adj_close", "volume", "series"]
    keep = [c for c in cols_order if c in df.columns]

    return df[keep]


def main():
    print("\n" + "="*80)
    print("[CLEAN] Macro Extra Cleaner (MOVE, VVIX, credit spreads, yield curve)")
    print("="*80)

    ensure_out_dir()

    for name, stype in SERIES_TYPES.items():
        print(f"\n[INFO] Cleaning {name} ({stype}) ...")

        df = load_file(name)
        if df is None:
            continue

        if stype == "fred_value":
            clean_df = clean_value_series(df, name)
        else:
            clean_df = clean_ohlcv(df, name)

        if clean_df is None or clean_df.empty:
            print(f"[WARN] Empty cleaned DF for {name}")
            continue

        out_path = os.path.join(OUT_DIR, f"{name}.parquet")
        clean_df.to_parquet(out_path, index=False)
        print(f"[OK] Saved cleaned {name} → {out_path} (rows={len(clean_df)})")

    print("\n[COMPLETE] Macro-extra cleaning finished.\n")


if __name__ == "__main__":
    main()
