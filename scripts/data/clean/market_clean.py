#!/usr/bin/env python3
"""
clean_market.py

Minimal safe cleaning for macro-market data (commodities, FX, DXY).

Rules:
- Keep NaN (market closed days)
- DO NOT fill or drop NaN
- Drop duplicate dates
- Ensure numeric dtype
- Preserve original OHLCV structure exactly
"""

import os
import yaml
import pandas as pd

CONFIG_PATH = "config/data.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date" not in df.columns:
        raise RuntimeError("No 'date' column found in macro-market frame.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="first")

    # Drop object columns if any
    obj_cols = df.select_dtypes(include=["object"]).columns
    obj_cols = [c for c in obj_cols if c != "date"]
    df = df.drop(columns=obj_cols)

    # Convert numeric
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    return df


def main():
    cfg = load_config()

    raw_dir = cfg["data"]["sources"]["market_folder"]
    out_dir = cfg["data"]["processed"]["market_folder"]
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(raw_dir) if f.endswith(".parquet")]
    print(f"[INFO] Found {len(files)} market raw files.")

    for f in files:
        df = clean_frame(pd.read_parquet(os.path.join(raw_dir, f)))
        df.to_parquet(os.path.join(out_dir, f), index=False)
        print(f"[OK] Cleaned → {os.path.join(out_dir, f)}")

    print("\n[COMPLETE] clean_market.py finished.\n")


if __name__ == "__main__":
    main()
