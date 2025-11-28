#!/usr/bin/env python3
"""
fetch_us_credit_spreads.py

Fetches US credit spreads from FRED:
- HY spread
- IG spread
- TED spread

LIBOR-OIS removed (not available on FRED).

Output files:
    data/raw_macro_extra/HY_SPREAD.parquet
    data/raw_macro_extra/IG_SPREAD.parquet
    data/raw_macro_extra/TED_SPREAD.parquet

Schema:
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
# Fetch a single FRED series
# ----------------------------------------------------------
def fetch_fred_series(fred: Fred, code: str, name: str, region: str, start: str):
    print(f"\n[INFO] Fetching {name} ({code}) ...")

    try:
        series = fred.get_series(code)
    except Exception as e:
        print(f"[ERROR] Failed to fetch {name}: {e}")
        return None

    if series is None:
        print(f"[WARN] Empty series: {name}")
        return None

    df = series.to_frame("value").reset_index()
    df.columns = ["date", "value"]

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= start]

    df["series"] = name
    df["region"] = region
    df["source"] = "FRED"

    return df


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    print("\n[INFO] Fetching US credit spreads from FRED ...")

    if os.getenv("FRED_API_KEY") is None:
        raise RuntimeError("Missing FRED_API_KEY. Please run: export FRED_API_KEY='...'")

    cfg = load_config()
    start = cfg["data"]["start_date"]
    region = "US"

    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    out_dir = "data/raw_macro_extra"
    os.makedirs(out_dir, exist_ok=True)

    # FRED codes
    SERIES = {
        "HY_SPREAD": "BAMLH0A0HYM2",
        "IG_SPREAD": "BAMLCC0A1AAATRIV",
        "TED_SPREAD": "TEDRATE"
    }

    for name, code in SERIES.items():
        df = fetch_fred_series(fred, code, name, region, start)
        if df is None or df.empty:
            continue

        out_path = os.path.join(out_dir, f"{name}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"[OK] Saved {name} → {out_path} (rows={len(df)})")

    print("\n[COMPLETE] US credit spreads fetch finished.\n")


if __name__ == "__main__":
    main()
