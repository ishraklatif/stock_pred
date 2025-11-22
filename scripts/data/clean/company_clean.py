#!/usr/bin/env python3
"""
clean_company.py

Cleans company datasets immediately after fetching:
- ensures 'date' exists and sorted
- drops object columns
- drops columns >50% NaN
- drops rows >30% NaN
- forward/back fill
- convert numeric to float32

Output → data/processed_companies/<ticker>.parquet
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

    # ensure date column
    if "date" not in df.columns:
        raise RuntimeError("No 'date' column found for company file.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")

    # drop object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    df = df.drop(columns=obj_cols)

    # drop columns with too many NaN
    df = df.dropna(axis=1, thresh=int(len(df) * 0.50))

    # drop rows with too many NaN
    df = df.dropna(axis=0, thresh=int(df.shape[1] * 0.70))

    # fill remaining
    df = df.ffill().bfill()

    # cast numeric
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

    for f in files:
        fpath = os.path.join(raw_dir, f)
        df = pd.read_parquet(fpath)
        df = clean_frame(df)

        out_path = os.path.join(out_dir, f)
        df.to_parquet(out_path, index=False)
        print(f"[OK] Cleaned → {out_path}")

    print("\n[COMPLETE] clean_company.py finished.\n")


if __name__ == "__main__":
    main()
