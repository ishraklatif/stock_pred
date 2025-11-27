#!/usr/bin/env python3
"""
fetch_fred_macro.py

Fetch US macroeconomic indicators from FRED API.

Inputs:
  - YAML config (data.yaml)
  - FRED series codes under fred_macro.sources

Outputs:
  - One parquet file per canonical series:
        data/raw_fred/US_CPI.parquet
        data/raw_fred/US_UNEMP.parquet
        ...

All files have columns:
    date, value, series, region, source
"""

import os
import yaml
import pandas as pd
from datetime import datetime
from fredapi import Fred

CONFIG_PATH = "config/data.yaml"


# ----------------------------------------------------------
# Load config
# ----------------------------------------------------------
def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    if os.getenv("FRED_API_KEY") is None:
        raise RuntimeError("Missing FRED_API_KEY. Please run: export FRED_API_KEY='...'")

    cfg = load_config()

    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or datetime.today().strftime("%Y-%m-%d")

    out_dir = cfg["data"]["sources"]["fred_folder"]
    os.makedirs(out_dir, exist_ok=True)

    region_prefix = cfg["fred_macro"]["canonical_prefix"]  # "US"
    fred_series = cfg["fred_macro"]["sources"]             # dict

    # Load API key (export FRED_API_KEY="...")
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    print(f"[INFO] Saving FRED macro → {out_dir}")

    for name, code in fred_series.items():
        print(f"[INFO] Fetching {name} ({code}) ...")

        try:
            series = fred.get_series(code)
        except Exception as e:
            print(f"[ERROR] Failed {name}: {e}")
            continue

        df = series.to_frame("value").reset_index()
        df.columns = ["date", "value"]

        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start)]

        df["series"] = name
        df["region"] = region_prefix
        df["source"] = "FRED"

        fname = f"{region_prefix}_{name}.parquet"
        out = os.path.join(out_dir, fname)

        df.to_parquet(out, index=False)
        print(f"[OK] Saved {out} ({len(df)} rows)")

    print("\n[COMPLETE] FRED macro fetch finished.\n")


if __name__ == "__main__":
    main()
