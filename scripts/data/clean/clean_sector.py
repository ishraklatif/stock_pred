#!/usr/bin/env python3
"""
clean_sector.py

Cleans sector ETF OHLCV data (XLF, XLK, XLE, ...).

Rules:
- Parse date
- Sort by date
- Drop duplicate dates
- Keep NaN (market holidays)
- Ensure numeric for OHLCV
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
        raise RuntimeError("No 'date' column found in sector frame.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="first")

    # Drop any stray object columns except date
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != "date"]
    df = df.drop(columns=obj_cols)

    # Cast numeric
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    return df


def main():
    cfg = load_config()

    raw_dir = cfg["data"]["sources"]["sector_folder"]
    out_dir = cfg["data"]["processed"]["sector_folder"]
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(raw_dir) if f.endswith(".parquet")]
    print(f"[INFO] Found {len(files)} sector raw files.")

    for fname in files:
        raw_path = os.path.join(raw_dir, fname)
        df = pd.read_parquet(raw_path)
        clean_df = clean_frame(df)

        out_path = os.path.join(out_dir, fname)
        clean_df.to_parquet(out_path, index=False)
        print(f"[OK] Cleaned sector → {out_path}")

    print("\n[COMPLETE] clean_sector.py finished.\n")


if __name__ == "__main__":
    main()
