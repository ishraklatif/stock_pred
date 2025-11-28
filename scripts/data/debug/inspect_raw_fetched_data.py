#!/usr/bin/env python3
"""
inspect_raw_fetched_data.py — UNIVERSAL + ABS DEEP INSPECTION VERSION

Inspects ALL raw fetched data before cleaning/merging.

New Features Added:
-------------------
ABS special handling:
- Detect SDMX format (TIME_PERIOD + OBS_VALUE)
- Detect canonical format (date + value)
- GDP candidate detection (files containing GDP)
- Terms of Trade detection (files containing TTR)
- Column validation, numeric counts, NaN diagnostics
- Head/tail preview for each ABS file
- Flag invalid ABS files before cleaning

Covers:
- data/raw_macro
- data/raw_macro_market
- data/raw_fred
- data/raw_rba
- data/raw_sector
- data/raw_companies
- data/raw_abs    (deep inspection added)
- data/news/raw

"""

import os
import json
import pandas as pd
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)

from scripts.data.fetch.canonical_map import (
    canonical_name,
    safe_filename,
    group_by_canonical,
)


# RAW FOLDERS TO INSPECT
RAW_FOLDERS = {
    "MACRO (Yahoo indices)": "data/raw_macro",
    "MARKET (Yahoo FX/Commodities)": "data/raw_macro_market",
    "FRED (US macro)": "data/raw_fred",
    "RBA (AU rates & yields)": "data/raw_rba",
    "SECTOR ETFs": "data/raw_sector",
    "COMPANIES": "data/raw_companies",
    "ABS (SDMX/Canonical mix)": "data/raw_abs",
}

NEWS_FOLDER = "data/news/raw"


# =============================================================================
# BASIC HELPERS
# =============================================================================
def load_file(path):
    """Load parquet or CSV safely."""
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


def describe_df(df):
    """Generic description for non-ABS data."""
    if df is None or df.empty:
        return "EMPTY"

    info = f"rows={len(df)}, cols={list(df.columns)}"

    if "date" in df.columns:
        try:
            d1 = str(df["date"].min())[:10]
            d2 = str(df["date"].max())[:10]
            info += f", date_range={d1} → {d2}"
        except:
            pass

    return info


def compare_dfs(df1, df2):
    if df1 is None or df2 is None:
        return False
    if list(df1.columns) != list(df2.columns):
        return False
    if len(df1) != len(df2):
        return False
    return df1.reset_index(drop=True).equals(df2.reset_index(drop=True))


# =============================================================================
# ABS-SPECIFIC INSPECTION
# =============================================================================
def detect_abs_schema(df):
    """
    Detect ABS schema type:
    - 'sdmx'      : TIME_PERIOD + OBS_VALUE
    - 'canonical' : date + value
    - 'unknown'
    """
    cols = set(df.columns)

    if "TIME_PERIOD" in cols and "OBS_VALUE" in cols:
        return "sdmx"

    if "date" in cols and "value" in cols:
        return "canonical"

    return "unknown"


def is_gdp_file(fname: str):
    return "GDP" in fname.upper() and "TTR" not in fname.upper()


def is_tot_file(fname: str):
    return "TTR" in fname.upper()


def inspect_abs_file(path):
    fname = os.path.basename(path)
    df = load_file(path)

    print(f"\n[ABS] {fname}")

    if df is None or df.empty:
        print("   -> EMPTY or unreadable")
        return

    schema = detect_abs_schema(df)
    print(f"   Schema: {schema}")

    cols = list(df.columns)
    print(f"   Columns: {cols}")

    # ----- Type classification -----
    type_tag = "-"
    if is_gdp_file(fname):
        type_tag = "GDP candidate"
    if is_tot_file(fname):
        type_tag = "ToT candidate"

    print(f"   Type detection: {type_tag}")

    # ----- Extract dates -----
    try:
        if schema == "sdmx":
            dates = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
        elif schema == "canonical":
            dates = pd.to_datetime(df["date"], errors="coerce")
        else:
            # Fallback: try first date-like column
            date_cols = [c for c in df.columns if "date" in c.lower() or "period" in c.lower()]
            dates = pd.to_datetime(df[date_cols[0]], errors="coerce") if date_cols else pd.Series([])

        d1 = str(dates.min())[:10]
        d2 = str(dates.max())[:10]
    except:
        d1, d2 = "NA", "NA"

    print(f"   Date range: {d1} → {d2}")

    # ----- Value stats -----
    try:
        if schema == "sdmx":
            vals = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        elif schema == "canonical":
            vals = pd.to_numeric(df["value"], errors="coerce")
        else:
            # Try fallback
            val_cols = [c for c in df.columns if c.lower() in ["value", "obs_value"]]
            vals = pd.to_numeric(df[val_cols[0]], errors="coerce") if val_cols else pd.Series([])

        numeric_count = vals.notna().sum()
        nan_count = vals.isna().sum()
    except:
        numeric_count = nan_count = "?"

    print(f"   Numeric: {numeric_count}, NaN: {nan_count}")

    # Print head/tail
    print("   Head:")
    print(df.head(3))
    print("   Tail:")
    print(df.tail(3))


# =============================================================================
# FOLDER INSPECTION LOGIC
# =============================================================================
def inspect_folder(name, folder_path):
    print("\n" + "=" * 80)
    print(f"INSPECTING: {name}")
    print("=" * 80)

    if not os.path.exists(folder_path):
        print(f"[WARN] Missing folder: {folder_path}")
        return

    files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".parquet") or f.endswith(".csv")
    ])

    if not files:
        print("[WARN] No data files.")
        return

    # ABS special case
    if "ABS" in name.upper():
        for f in files:
            inspect_abs_file(os.path.join(folder_path, f))
        return

    # Other folders → use canonical grouping logic
    symbol_paths = {
        os.path.splitext(f)[0]: os.path.join(folder_path, f)
        for f in files
    }

    grouped = group_by_canonical(symbol_paths.keys())

    for canon, raw_syms in grouped.items():
        print(f"\n--- CANONICAL: {canon} ---")
        print("Raw symbols:", raw_syms)

        dfs = {}

        for sym in raw_syms:
            path = symbol_paths.get(sym)
            df = load_file(path)
            dfs[sym] = df

            print(f"  [{sym}] {describe_df(df)}")

        # Compare alternates
        syms = list(dfs.keys())
        if len(syms) > 1:
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    same = compare_dfs(dfs[syms[i]], dfs[syms[j]])
                    status = "IDENTICAL" if same else "DIFFERENT"
                    print(f"    [{status}] {syms[i]} vs {syms[j]}")


def inspect_news():
    print("\n" + "=" * 80)
    print("INSPECTING: NEWS RAW JSON")
    print("=" * 80)

    if not os.path.exists(NEWS_FOLDER):
        print("[WARN] News folder missing.")
        return

    files = [f for f in os.listdir(NEWS_FOLDER) if f.endswith(".json")]

    if not files:
        print("[WARN] No news files.")
        return

    for f in files:
        try:
            data = json.load(open(os.path.join(NEWS_FOLDER, f)))
            print(f"[{f}] count={len(data)}")
        except Exception as e:
            print(f"[ERROR] {f}: {e}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    for name, folder in RAW_FOLDERS.items():
        inspect_folder(name, folder)

    inspect_news()

    print("\n" + "=" * 80)
    print("RAW FETCH INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
