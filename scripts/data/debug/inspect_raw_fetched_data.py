#!/usr/bin/env python3
"""
inspect_raw_fetched_data.py — UNIVERSAL VERSION

Inspects ALL raw fetched data files before cleaning/merging.

Covers:
- data/raw_macro              (Yahoo global indices)
- data/raw_macro_market       (Yahoo commodities/FX)
- data/raw_fred               (US macro from FRED)
- data/raw_rba                (AU macro RBA tables)
- data/raw_sector             (Sector ETFs)
- data/raw_companies          (company OHLCV)
- data/raw_abs                (ABS future)
- data/news/raw               (news JSON)

This script:
- Detects empty/malformed files
- Compares alternates under canonical_map
- Describes columns, shape, date range
- Works for both OHLCV and macro CSV/Parquet
"""

import os
import json
import pandas as pd
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)

from scripts.data.fetch.canonical_map import (
    canonical_name,
    safe_filename,
    group_by_canonical,
)

# RAW FOLDERS
RAW_FOLDERS = {
    "MACRO (Yahoo indices)": "data/raw_macro",
    "MARKET (Yahoo FX/Commodities)": "data/raw_macro_market",
    "FRED (US macro)": "data/raw_fred",
    "RBA (AU rates & yields)": "data/raw_rba",
    "SECTOR ETFs": "data/raw_sector",
    "COMPANIES": "data/raw_companies",
    "ABS (future)": "data/raw_abs",
}


NEWS_FOLDER = "data/news/raw"


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def load_file(path):
    """Load either parquet or CSV."""
    try:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        elif path.endswith(".csv"):
            return pd.read_csv(path)
        else:
            return None
    except Exception as e:
        print(f"[ERROR] Could not load {path}: {e}")
        return None


def compare_dfs(df1, df2):
    """Check if two OHLCV or macro dataframes are identical."""
    if df1 is None or df2 is None:
        return False
    if list(df1.columns) != list(df2.columns):
        return False
    if len(df1) != len(df2):
        return False
    return df1.reset_index(drop=True).equals(df2.reset_index(drop=True))


def describe_df(df):
    """Print basic info about a dataframe."""
    if df is None or df.empty:
        return "EMPTY"

    info = f"rows={len(df)}, cols={list(df.columns)}"
    # try date range
    if "date" in df.columns:
        try:
            d1 = str(df['date'].min())[:10]
            d2 = str(df['date'].max())[:10]
            info += f", date_range={d1} → {d2}"
        except:
            pass
    return info


# -------------------------------------------------------------------
# INSPECTION LOGIC
# -------------------------------------------------------------------
def inspect_folder(name, folder_path):
    print("\n" + "=" * 80)
    print(f" INSPECTING: {name}")
    print("=" * 80)

    if not os.path.exists(folder_path):
        print(f"[WARN] Folder missing: {folder_path}")
        return

    files = [f for f in os.listdir(folder_path)
             if f.endswith(".parquet") or f.endswith(".csv")]

    if not files:
        print("[WARN] No data files found.")
        return

    # Map raw file symbol → path
    symbol_paths = {
        os.path.splitext(f)[0]: os.path.join(folder_path, f)
        for f in files
    }

    # Group by canonical
    grouped = group_by_canonical(symbol_paths.keys())

    for canon, raw_syms in grouped.items():
        print(f"\n--- CANONICAL: {canon} ---")
        print("Raw symbols:", raw_syms)

        dfs = {}

        # Load each raw file
        for sym in raw_syms:
            path = symbol_paths.get(sym)
            if not path:
                print(f"  [SKIP] Missing file for symbol {sym}")
                continue

            df = load_file(path)
            dfs[sym] = df

            if df is None or df.empty:
                print(f"  [{sym}] EMPTY / invalid → {path}")
            else:
                print(f"  [{sym}] {describe_df(df)}")

        # Compare alternates
        syms = list(dfs.keys())
        if len(syms) > 1:
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    s1, s2 = syms[i], syms[j]
                    same = compare_dfs(dfs[s1], dfs[s2])
                    tag = "IDENTICAL" if same else "DIFFERENT"
                    print(f"    [{tag}] {s1} vs {s2}")


def inspect_news():
    print("\n" + "=" * 80)
    print(" INSPECTING: NEWS RAW JSON")
    print("=" * 80)

    if not os.path.exists(NEWS_FOLDER):
        print("[WARN] news folder missing.")
        return

    files = [f for f in os.listdir(NEWS_FOLDER) if f.endswith(".json")]
    if not files:
        print("[WARN] No news files.")
        return

    for f in files:
        path = os.path.join(NEWS_FOLDER, f)
        try:
            with open(path, "r") as fp:
                data = json.load(fp)
            print(f"[{f}] count={len(data)}")
        except Exception as e:
            print(f"[ERROR] {f}: {e}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    for name, folder in RAW_FOLDERS.items():
        inspect_folder(name, folder)

    inspect_news()

    print("\n" + "=" * 80)
    print(" RAW FETCH INSPECTION COMPLETE ")
    print("=" * 80)


if __name__ == "__main__":
    main()
# python3 -m scripts.data.debug.inspect_raw_fetched_data
