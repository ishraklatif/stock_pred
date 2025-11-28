#!/usr/bin/env python3
"""
clean_rba.py

Cleans RBA series:

- AUS_RBA_CASH_RATE.parquet
- AUS_RBA_YIELD_10Y.parquet
- AUS_RBA_YIELD_2Y.parquet
- AUS_CREDIT_A.parquet
- AUS_CREDIT_BBB.parquet
- AUS_CREDIT_SPREAD_A.parquet
- AUS_CREDIT_SPREAD_BBB.parquet

Input schemas (per file):

1) Tidy value series (F1/F2, F3 yields):
    date, value, series, region, source

2) Tidy spread series (F3 credit spreads):
    date, spread, series, region, source

Output schema (all files):
    date, value, series, region, source
"""

import os
import yaml
import pandas as pd

CONFIG_PATH = "config/data.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def clean_rba_file(df: pd.DataFrame, fname: str) -> pd.DataFrame | None:
    if "date" not in df.columns:
        print(f"[WARN] RBA file {fname} has no 'date' column, skipping.")
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # Normalize numeric column to 'value'
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    elif "spread" in df.columns:
        df["value"] = pd.to_numeric(df["spread"], errors="coerce")
        df = df.drop(columns=["spread"])
    else:
        # no obvious numeric column
        num_cols = [c for c in df.columns if c not in ("date", "series", "region", "source")]
        if not num_cols:
            print(f"[WARN] RBA file {fname} has no numeric column, skipping.")
            return None
        col = num_cols[0]
        df["value"] = pd.to_numeric(df[col], errors="coerce")
        df = df.drop(columns=[col])

    # Ensure series / region / source
    series_default = fname.replace(".parquet", "")
    if "series" not in df.columns:
        df["series"] = series_default
    df["series"] = df["series"].astype(str)

    if "region" not in df.columns:
        df["region"] = "AUS"
    if "source" not in df.columns:
        df["source"] = "RBA"

    df = df[["date", "value", "series", "region", "source"]]
    df = df.dropna(subset=["value"])

    return df


def main():
    cfg = load_config()

    raw_dir = cfg["data"]["sources"]["rba_folder"]
    out_dir = cfg["data"]["processed"]["rba_folder"]
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(raw_dir) if f.endswith(".parquet")]
    print(f"[INFO] Found {len(files)} raw RBA files.")

    for fname in files:
        raw_path = os.path.join(raw_dir, fname)
        df = pd.read_parquet(raw_path)

        clean_df = clean_rba_file(df, fname)
        if clean_df is None or clean_df.empty:
            continue

        out_path = os.path.join(out_dir, fname)
        clean_df.to_parquet(out_path, index=False)
        print(f"[OK] Cleaned RBA → {out_path}")

    print("\n[COMPLETE] clean_rba.py finished\n")


if __name__ == "__main__":
    main()
