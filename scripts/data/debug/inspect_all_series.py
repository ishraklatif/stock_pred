#!/usr/bin/env python3
"""
inspect_all_series.py

Goal:
    Inspect ALL processed data files so compute_regimes.py can be written
    with ZERO assumptions.

Covers:
    data/processed_macro/
    data/processed_macro_market/
    data/processed_macro_extra/
    data/processed_rba/
    data/processed_fred/
    data/processed_sector/
    data/processed_companies/
    data/processed_abs/
    data/processed_abs/abs_all_series/   (if exists)

Outputs:
    data/inspection/schema_report.json
    data/inspection/schema_report.txt

Each entry includes:
    - file path
    - columns
    - dtype per column
    - null counts
    - date min / date max
    - inferred frequency (daily, monthly, quarterly, irregular)
    - file_type classification:
         * "ohlc" (Yahoo OHLCV)
         * "value_series" (FRED/ABS/RBA)
         * "unknown"
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

print("[DEBUG] ROOT DIR:", ROOT)
print("[DEBUG] ROOT/data exists?", (ROOT / "data").exists())
print("[DEBUG] ROOT/scripts exists?", (ROOT / "scripts").exists())


DIRS_TO_SCAN = [
    ROOT / "data/processed_macro",
    ROOT / "data/processed_macro_market",
    ROOT / "data/processed_macro_extra",
    ROOT / "data/processed_rba",
    ROOT / "data/processed_fred",
    ROOT / "data/processed_sector",
    ROOT / "data/processed_companies",
    ROOT / "data/processed_abs",
    ROOT / "data/processed_abs" / "abs_all_series",   # only if exists
]

OUT_DIR = ROOT / "data/inspection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUT = OUT_DIR / "schema_report.json"
TXT_OUT  = OUT_DIR / "schema_report.txt"


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def infer_frequency(date_series: pd.Series) -> str:
    """Infer calendar frequency: daily, monthly, quarterly, irregular."""
    if date_series.empty:
        return "empty"

    ds = date_series.sort_values().dropna()
    if len(ds) < 5:
        return "too_small"

    diffs = ds.diff().dt.days.dropna()
    median_gap = diffs.median()

    if median_gap <= 2:
        return "daily"
    elif 25 <= median_gap <= 35:
        return "monthly"
    elif 80 <= median_gap <= 100:
        return "quarterly"
    else:
        return f"irregular (median_gap={median_gap})"


def classify_schema(df: pd.DataFrame) -> str:
    """
    Determine file type based on columns.
    """
    cols = set(df.columns)

    # Yahoo-style OHLC
    if {"open", "high", "low", "close"}.issubset(cols):
        return "ohlc"

    # FRED/RBA/ABS typical structure
    if {"date", "value"}.issubset(cols):
        return "value_series"

    if {"TIME_PERIOD", "OBS_VALUE"}.issubset(cols):
        return "sdmx_raw"

    return "unknown"


def describe_file(path: Path) -> dict:
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {
            "file": str(path),
            "error": f"Cannot read parquet: {e}"
        }

    # Check date column
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break

    if date_col is None:
        date_min = date_max = None
        freq = "no_date_col"
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        date_min = str(df[date_col].min())
        date_max = str(df[date_col].max())
        freq = infer_frequency(df[date_col])

    return {
        "file": str(path),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "n_rows": int(len(df)),
        "date_column": date_col,
        "date_min": date_min,
        "date_max": date_max,
        "frequency": freq,
        "classification": classify_schema(df)
    }


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    report = []

    for directory in DIRS_TO_SCAN:
        if not directory.exists():
            continue

        print(f"\nInspecting directory: {directory}")
        for fp in sorted(directory.glob("*.parquet")):
            print(f"  -> File: {fp.name}")
            info = describe_file(fp)
            report.append(info)

    # Write JSON
    with open(JSON_OUT, "w") as f:
        json.dump(report, f, indent=4)

    # Pretty text report
    with open(TXT_OUT, "w") as f:
        for item in report:
            f.write("\n" + "="*80 + "\n")
            f.write(f"FILE: {item['file']}\n")
            f.write(f"TYPE: {item['classification']}\n")
            f.write(f"ROWS: {item['n_rows']}\n")
            f.write(f"DATE COL: {item['date_column']}\n")
            f.write(f"DATE RANGE: {item['date_min']} → {item['date_max']}\n")
            f.write(f"FREQUENCY: {item['frequency']}\n")
            f.write("COLUMNS:\n")
            for c in item["columns"]:
                f.write(f"  - {c} ({item['dtypes'][c]}, nulls={item['null_counts'][c]})\n")

    print("\n" + "="*80)
    print(f"[DONE] Schema report saved:\n- {JSON_OUT}\n- {TXT_OUT}")
    print("="*80)


if __name__ == "__main__":
    main()
