#!/usr/bin/env python3
"""
clean_macro.py

Minimal safe cleaning for macro index OHLCV data.

Rules:
- Never forward-fill or backward-fill macro values
- Keep NaN (market holidays cross-country alignment)
- Drop duplicate dates only
- Ensure numeric dtype
- DO NOT drop rows due to NaN
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
        raise RuntimeError("No 'date' column found in macro frame.")

    # Clean date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # Drop duplicate dates
    df = df.drop_duplicates(subset=["date"], keep="first")

    # Drop object columns (rare)
    obj_cols = df.select_dtypes(include=["object"]).columns
    obj_cols = [c for c in obj_cols if c != "date"]
    df = df.drop(columns=obj_cols)

    # Cast numeric
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # Keep NaN — DO NOT ffill/bfill
    return df


def main():
    cfg = load_config()

    raw_dir = cfg["data"]["sources"]["macro_folder"]
    out_dir = cfg["data"]["processed"]["macro_folder"]
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(raw_dir) if f.endswith(".parquet")]
    print(f"[INFO] Found {len(files)} macro raw files.")

    for f in files:
        df = clean_frame(pd.read_parquet(os.path.join(raw_dir, f)))
        df.to_parquet(os.path.join(out_dir, f), index=False)
        print(f"[OK] Cleaned → {os.path.join(out_dir, f)}")

    print("\n[COMPLETE] clean_macro.py finished.\n")


if __name__ == "__main__":
    main()
