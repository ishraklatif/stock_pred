#!/usr/bin/env python3
"""
extract_abs_series.py

Extract ALL usable ABS time-series, including:
- SDMX-style raw files (TIME_PERIOD, OBS_VALUE, MEASURE…)
- Already-cleaned ABS files (date, value, series, region, source)

Input:
    data/raw_abs/skipped/*.parquet

Output:
    data/processed_abs/abs_all_series/*.parquet

Output schema ALWAYS:
    date, value, series, region, source
"""

import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SKIPPED_DIR = Path("data/raw_abs/skipped")
OUT_DIR     = Path("data/processed_abs/abs_all_series")

os.makedirs(OUT_DIR, exist_ok=True)

# Dimensions to try for grouping SDMX tables
DIM_CANDIDATES = [
    "MEASURE", "INDEX", "TSEST", "REGION", "FREQ",
    "PROPERTY_TYPE", "SEX", "AGE", "SECTOR", "INDUSTRY"
]


def load_df(path: Path):
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"[ERROR] {path.name}: cannot load ({e})")
        return None


def is_clean_schema(df: pd.DataFrame):
    """Detect already cleaned ABS files."""
    needed = {"date", "value", "series", "region", "source"}
    return needed.issubset(df.columns)


def is_sdmx_schema(df: pd.DataFrame):
    """Detect SDMX raw tables."""
    return ("TIME_PERIOD" in df.columns) and ("OBS_VALUE" in df.columns)


def clean_existing_timeseries(df: pd.DataFrame, file_stem: str) -> pd.DataFrame:
    """
    Already-cleaned ABS file:
        date, value, series, region, source
    Just ensure sorting + consistency.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")

    # Ensure correct column order
    out = out[["date", "value", "series", "region", "source"]]
    return out


def clean_sdmx_group(df: pd.DataFrame, series_name: str) -> pd.DataFrame:
    """
    Convert SDMX grouped subset to canonical time-series.
    """
    out = df.copy()

    out["date"] = pd.to_datetime(out["TIME_PERIOD"], errors="coerce")
    out["value"] = pd.to_numeric(out["OBS_VALUE"], errors="coerce")

    out = out.dropna(subset=["date", "value"]).sort_values("date")

    out["series"] = series_name
    out["region"] = "AUS"
    out["source"] = "ABS"

    return out[["date", "value", "series", "region", "source"]]


def detect_dims(df: pd.DataFrame):
    return [c for c in DIM_CANDIDATES if c in df.columns]


def make_series_name(file_stem, dim_cols, key_vals):
    if not isinstance(key_vals, tuple):
        key_vals = (key_vals,)
    parts = []
    for col, val in zip(dim_cols, key_vals):
        sval = str(val).replace(" ", "")
        parts.append(f"{col}{sval}")
    suffix = "_".join(parts)
    return f"{file_stem}__{suffix}"


def process_file(path: Path) -> int:
    df = load_df(path)
    if df is None or df.empty:
        print(f"[WARN] {path.name}: empty")
        return 0

    file_stem = path.stem

    # ---------------------------------------------------------
    # CASE 1: Already-cleaned ABS file
    # ---------------------------------------------------------
    if is_clean_schema(df):
        try:
            cleaned = clean_existing_timeseries(df, file_stem)
            out_path = OUT_DIR / f"{file_stem}.parquet"
            cleaned.to_parquet(out_path, index=False)
            print(f"[OK] {file_stem}: saved 1 cleaned series → {out_path}")
            return 1
        except Exception as e:
            print(f"[ERROR] Cannot clean existing ABS file {file_stem}: {e}")
            return 0

    # ---------------------------------------------------------
    # CASE 2: SDMX raw file (never cleaned)
    # ---------------------------------------------------------
    if is_sdmx_schema(df):
        dim_cols = detect_dims(df)

        # If no dimension columns → treat whole file as ONE series
        if not dim_cols:
            sname = file_stem
            ts = clean_sdmx_group(df, sname)
            out_path = OUT_DIR / f"{sname}.parquet"
            ts.to_parquet(out_path, index=False)
            print(f"[OK] {file_stem}: wrote single SDMX series → {out_path}")
            return 1

        # Group SDMX tables by dimensions → multi-series extraction
        n = 0
        grouped = df.groupby(dim_cols, dropna=False)

        for key_vals, subdf in grouped:
            sname = make_series_name(file_stem, dim_cols, key_vals)

            try:
                ts = clean_sdmx_group(subdf, sname)
                if not ts.empty:
                    ts.to_parquet(OUT_DIR / f"{sname}.parquet", index=False)
                    n += 1
            except Exception as e:
                print(f"[ERROR] {file_stem} group failed {sname}: {e}")

        print(f"[OK] {file_stem}: extracted {n} SDMX series")
        return n

    # ---------------------------------------------------------
    # CASE 3: Unknown format → dump file meta
    # ---------------------------------------------------------
    print(f"[WARN] {file_stem}: unknown schema (columns={list(df.columns)})")
    return 0


def main():
    print("\n" + "=" * 80)
    print("[ABS] Extract ALL time-series from skipped ABS files")
    print("=" * 80)

    if not SKIPPED_DIR.exists():
        print("[ERROR] Skipped folder missing. Run clean_abs first.")
        return

    files = sorted(SKIPPED_DIR.glob("*.parquet"))
    if not files:
        print("[INFO] No files in skipped/")
        return

    total = 0
    for fp in files:
        print("\n" + "-" * 80)
        print(f"[FILE] {fp.name}")
        print("-" * 80)
        total += process_file(fp)

    print("\n" + "=" * 80)
    print(f"[COMPLETE] Extracted total series: {total}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
