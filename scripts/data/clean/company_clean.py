#!/usr/bin/env python3
"""
clean_company.py — canonical-aware version

Loads raw_company/*.parquet using safe_ticker_name,
cleans them, and writes them into processed_companies/*.parquet
"""

import os
import yaml
import pandas as pd

from scripts.data.fetch.canonical_map import safe_filename

CONFIG_PATH = "config/data.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "date" not in df.columns:
        raise RuntimeError("Missing 'date' column in company data")

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")

    # Drop text/object columns
    df = df.drop(columns=df.select_dtypes(include=["object"]).columns)

    # Drop columns with >50% NaN
    df = df.dropna(axis=1, thresh=int(len(df) * 0.5))

    # Drop rows with >30% NaN
    df = df.dropna(axis=0, thresh=int(df.shape[1] * 0.7))

    # Forward/backfill
    df = df.ffill().bfill()

    # Cast numeric
    for col in df.columns:
        if col != "date":
            df[col] = df[col].astype("float32")

    return df


def main():
    cfg = load_config()

    raw_dir = cfg["data"]["sources"]["companies_folder"]
    out_dir = cfg["data"]["processed"]["companies_folder"]
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(raw_dir) if f.endswith(".parquet")]
    print(f"[INFO] Found {len(files)} company raw files.")

    for fname in files:
        raw_path = os.path.join(raw_dir, fname)

        raw_name = fname.replace(".parquet", "")
        safe = safe_filename(raw_name)

        df = pd.read_parquet(raw_path)
        df = clean_frame(df)

        out_path = os.path.join(out_dir, f"{safe}.parquet")
        df.to_parquet(out_path, index=False)

        print(f"[OK] Cleaned → {out_path}")

    print("\n[COMPLETE] clean_company.py finished.\n")


if __name__ == "__main__":
    main()
