#!/usr/bin/env python3
"""
inspect_raw_fetched_data.py

Diagnostic tool to inspect raw fetched data (before clean/compute/merge).
Useful to identify:
- Duplicate assets before merging
- Multiple alternate symbols mapping to the same canonical
- Conflicting OHLCV data across alternates
- Identical data duplicates (safe to drop)
- Empty / malformed raw files

Works entirely on RAW FILES:
    data/raw_macro/
    data/raw_macro_market/
    data/news/raw/

Run:
    python scripts/debug/inspect_raw_fetched_data.py
"""

import os
import json
import pandas as pd
import sys

# Add project root so "scripts" becomes importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)

from scripts.data.fetch.canonical_map import (
    canonical_name,
    safe_filename,
    group_by_canonical,
)



RAW_MACRO = "data/raw_macro"
RAW_MARKET = "data/raw_macro_market"
RAW_NEWS = "data/news/raw"


def load_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"[ERROR] Could not load {path}: {e}")
        return None


def compare_dfs(df1, df2):
    """Check if two OHLCV dataframes are identical."""
    if df1 is None or df2 is None:
        return False
    if list(df1.columns) != list(df2.columns):
        return False
    if len(df1) != len(df2):
        return False
    return df1.reset_index(drop=True).equals(df2.reset_index(drop=True))


def inspect_folder(folder_name):
    print("\n====================================================")
    print(f" RAW FOLDER INSPECTION: {folder_name}")
    print("====================================================")

    files = [f for f in os.listdir(folder_name) if f.endswith(".parquet")]
    if not files:
        print("[WARN] No files found.")
        return

    symbol_to_path = {}
    for f in files:
        raw_symbol = os.path.splitext(f)[0]  # filename without extension
        symbol_to_path[raw_symbol] = os.path.join(folder_name, f)

    # Group by canonical name
    grouped = group_by_canonical(symbol_to_path.keys())

    # For each canonical instrument, load and inspect all alternates
    for canon, raw_symbols in grouped.items():
        print(f"\n--- CANONICAL: {canon} ---")
        print(f"Raw symbols: {raw_symbols}")

        dfs = {}
        for sym in raw_symbols:
            path = symbol_to_path.get(sym)
            if path is None:
                print(f"  [SKIP] No file for symbol {sym}")
                continue

            df = load_parquet(path)
            if df is None or df.empty:
                print(f"  [WARN] EMPTY or invalid data: {path}")
            dfs[sym] = df

            # Basic shape / column checks
            if df is not None:
                print(f"  [{sym}] shape={df.shape}, columns={list(df.columns)}")

        # Compare alternates internally
        syms = list(dfs.keys())
        if len(syms) > 1:
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    s1, s2 = syms[i], syms[j]
                    same = compare_dfs(dfs[s1], dfs[s2])
                    if same:
                        print(f"    [OK] {s1} == {s2} (identical datasets)")
                    else:
                        print(f"    [DIFF] {s1} != {s2} (different data!)")


def inspect_news():
    print("\n====================================================")
    print(" RAW NEWS INSPECTION ")
    print("====================================================")

    if not os.path.exists(RAW_NEWS):
        print("[WARN] news folder does not exist")
        return

    files = [f for f in os.listdir(RAW_NEWS) if f.endswith(".json")]
    if not files:
        print("[WARN] No news JSON files found.")
        return

    for f in files:
        fpath = os.path.join(RAW_NEWS, f)
        canon = os.path.splitext(f)[0]

        try:
            with open(fpath, "r") as fp:
                data = json.load(fp)
        except Exception as e:
            print(f"[ERROR] Could not load {fpath}: {e}")
            continue

        print(f"[{canon}] news_count={len(data)}")


def main():
    inspect_folder(RAW_MACRO)
    inspect_folder(RAW_MARKET)
    inspect_news()

    print("\n====================================================")
    print(" RAW FETCH INSPECTION COMPLETE ")
    print("====================================================")


if __name__ == "__main__":
    main()

