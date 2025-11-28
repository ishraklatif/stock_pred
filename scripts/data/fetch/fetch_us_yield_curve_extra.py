#!/usr/bin/env python3
"""
fetch_us_yield_curve_extra.py

Fetch additional US yield curve points from FRED:

    US3M  → "DGS3MO"  (3-month Treasury)
    US5Y  → "DGS5"    (5-year Treasury)
    US30Y → "DGS30"   (30-year Treasury)

Output (per series):
    data/raw_macro_extra/US3M.parquet
    data/raw_macro_extra/US5Y.parquet
    data/raw_macro_extra/US30Y.parquet

Schema:
    date, value, series, region, source
"""

import os
from datetime import datetime

import yaml
import pandas as pd
from fredapi import Fred

CONFIG_PATH = "config/data.yaml"
DEFAULT_OUT_DIR = "data/raw_macro_extra"


FRED_YIELD_MAP = {
    "US3M": "DGS3MO",
    "US5Y": "DGS5",
    "US30Y": "DGS30",
}


# ----------------------------------------------------------
# Config helpers
# ----------------------------------------------------------
def load_config(path: str = CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        print(f"[WARN] Config file not found at {path}, using defaults.")
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_data_dates(cfg: dict) -> tuple[str, str]:
    data_cfg = cfg.get("data", {})
    start = data_cfg.get("start_date", "2000-01-01")
    end = data_cfg.get("end_date") or datetime.today().strftime("%Y-%m-%d")
    return start, end


def get_out_dir(cfg: dict) -> str:
    sources = cfg.get("data", {}).get("sources", {})
    return sources.get("macro_extra_folder", DEFAULT_OUT_DIR)


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    if os.getenv("FRED_API_KEY") is None:
        raise RuntimeError(
            "Missing FRED_API_KEY. Please run: export FRED_API_KEY='...'"
        )

    cfg = load_config()
    start, end = get_data_dates(cfg)
    out_dir = get_out_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)

    region_prefix = cfg.get("fred_macro", {}).get("canonical_prefix", "US")
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    print("[INFO] Fetching extra US yield curve points from FRED ...")

    for canonical, fred_code in FRED_YIELD_MAP.items():
        print(f"\n[INFO] Fetching {canonical} ({fred_code}) ...")
        try:
            series = fred.get_series(fred_code)
        except Exception as e:
            print(f"[ERROR] Failed to fetch {canonical}: {e}")
            continue

        df = series.to_frame("value").reset_index()
        df.columns = ["date", "value"]

        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]

        df = df.sort_values("date")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["value"] = df["value"].ffill()

        df = df.dropna(subset=["date"])

        if df.empty:
            print(f"[WARN] {canonical} DataFrame is empty after cleaning")
            continue

        df["series"] = canonical
        df["region"] = region_prefix
        df["source"] = "FRED"

        out_path = os.path.join(out_dir, f"{canonical}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"[OK] Saved {canonical} → {out_path} (rows={len(df)})")

    print("\n[COMPLETE] US yield curve extra fetch finished.\n")


if __name__ == "__main__":
    main()
