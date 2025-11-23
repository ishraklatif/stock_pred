#!/usr/bin/env python3
"""
fetch_macro_market.py

Fetches macro-market instruments such as:
DXY, FX pairs (AUDUSD, AUDJPY), commodities (GOLD, OIL, COPPER, SILVER), DBC.

Now with:
- Canonical naming via canonical_map.py
- Support for multiple alternate tickers unified to the same canonical
- Only ONE parquet per canonical asset (no duplicates)
"""

import os
from datetime import datetime
from typing import Optional, Tuple, List

import yaml
import yfinance as yf
import pandas as pd

from scripts.data.fetch.canonical_map import (
    canonical_name,
    safe_filename,
    group_by_canonical,
)

CONFIG_PATH = "config/data.yaml"


# ---------------------------------------------------------------------
# CONFIG LOADER
# ---------------------------------------------------------------------
def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# CLEAN YFINANCE DATAFRAME
# ---------------------------------------------------------------------
def clean_yf(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    flat_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            col = col[0]
        flat_cols.append(str(col).lower().replace(" ", "_"))
    df.columns = flat_cols

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")

    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    return df[[c for c in keep if c in df.columns]]


# ---------------------------------------------------------------------
# FETCH SINGLE SYMBOL
# ---------------------------------------------------------------------
def fetch_symbol(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    print(f"[INFO] Fetching macro-market: {symbol}")
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
    except Exception as e:
        print(f"[ERROR] yfinance error for {symbol}: {e}")
        return None

    if df is None or df.empty:
        print(f"[WARN] No data for {symbol}")
        return None

    return clean_yf(df)


# ---------------------------------------------------------------------
# CHOOSE BEST DF AMONG ALTERNATES
# ---------------------------------------------------------------------
def choose_best_df(
    candidates: List[Tuple[str, pd.DataFrame]]
) -> Tuple[str, pd.DataFrame]:
    """
    Given a list of (symbol, df) for the same canonical asset,
    pick the "best" one.

    Heuristic:
    - Prefer the dataframe with the longest history (most rows)
    - Tie-breaker: earliest start date
    """
    if not candidates:
        raise ValueError("No candidates provided to choose_best_df")

    def score(item: Tuple[str, pd.DataFrame]):
        sym, df = item
        n = len(df)
        start_date = df["date"].min() if "date" in df.columns else pd.Timestamp.max
        return (n, -start_date.timestamp())

    best_sym, best_df = max(candidates, key=score)
    return best_sym, best_df


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    cfg = load_config()

    raw_tickers = cfg["market"]["tickers"]  # e.g. ["DX-Y.NYB", "AUDUSD=X", "HG=F", ...]
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or datetime.today().strftime("%Y-%m-%d")

    out_dir = cfg["data"]["sources"]["market_folder"]
    os.makedirs(out_dir, exist_ok=True)

    # Group tickers by canonical instrument name
    grouped = group_by_canonical(raw_tickers)

    print(f"[INFO] Macro-market fetch – canonical groups: {grouped}")

    for canon, symbols in grouped.items():
        candidates: List[Tuple[str, pd.DataFrame]] = []

        for symbol in symbols:
            df = fetch_symbol(symbol, start, end)
            if df is None or df.empty:
                continue
            candidates.append((symbol, df))

        if not candidates:
            print(f"[WARN] No data for canonical asset {canon} (symbols={symbols})")
            continue

        best_sym, best_df = choose_best_df(candidates)
        fname = safe_filename(canon) + ".parquet"
        out = os.path.join(out_dir, fname)

        best_df.to_parquet(out, index=False)
        print(
            f"[OK] Saved canonical macro-market → {out} "
            f"(chosen from {symbols}, best='{best_sym}', rows={len(best_df)})"
        )

    print("\n[COMPLETE] Macro-market fetch done.\n")


if __name__ == "__main__":
    main()
