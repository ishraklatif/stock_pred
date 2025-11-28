#!/usr/bin/env python3
"""
clean_fred.py

Cleans US FRED macro series:
    CPI, UNEMP, FEDFUNDS, DGS10, DGS2, PMI, GDP, etc.

Output:
    date, value, series, region, source
"""

import os
import yaml
import pandas as pd

CONFIG_PATH = "config/data.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def clean_fred(df, series_name):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series"] = series_name
    df["region"] = "US"
    df["source"] = "FRED"

    # FRED data is usually daily but with gaps → ffill
    df = df.ffill()

    return df[["date", "value", "series", "region", "source"]]


def main():
    cfg = load_config()

    raw_dir = cfg["data"]["sources"]["fred_folder"]
    out_dir = cfg["data"]["processed"]["fred_folder"]
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(raw_dir) if f.endswith(".parquet")]

    print(f"[INFO] Found {len(files)} raw FRED files.")

    for fname in files:
        df = pd.read_parquet(os.path.join(raw_dir, fname))
        series_name = fname.replace(".parquet", "")

        cleaned = clean_fred(df, series_name)
        out_path = os.path.join(out_dir, fname)
        cleaned.to_parquet(out_path, index=False)

        print(f"[OK] Cleaned FRED → {out_path}")

    print("\n[COMPLETE] clean_fred.py finished\n")


if __name__ == "__main__":
    main()
