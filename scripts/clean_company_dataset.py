#!/usr/bin/env python3
"""
clean_company_dataset.py

Cleans all company merged datasets so they follow the same rules
as the macro_clean dataset.
"""

import os
import pandas as pd

INPUT_DIR = "data/level4_company"
OUTPUT_DIR = "data/level4_final/company"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_company(df: pd.DataFrame) -> pd.DataFrame:

    # FIX 1: restore Date column if it's the index
    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            print("  - Restoring Date column from index")
            df = df.reset_index()
        else:
            raise ValueError("Dataset has no Date column.")

    # Ensure proper datetime + sorting
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"])

    # FIX 2: drop ALL datetime-like columns except Date
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
    datetime_cols = [c for c in datetime_cols if c != "Date"]

    if datetime_cols:
        print(f"  - Dropping extra datetime columns: {datetime_cols}")
        df = df.drop(columns=datetime_cols)

    # Drop object/string columns
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        print(f"  - Dropping object/string columns: {object_cols}")
        df = df.drop(columns=object_cols)

    # Drop columns with >50% NaN
    col_thresh = int(len(df) * 0.50)
    before_cols = df.shape[1]
    df = df.dropna(axis=1, thresh=col_thresh)
    after_cols = df.shape[1]
    print(f"  - Dropped {before_cols - after_cols} columns (>50% NaN)")

    # Drop rows with >30% NaN
    row_thresh = int(df.shape[1] * 0.70)
    before_rows = df.shape[0]
    df = df.dropna(axis=0, thresh=row_thresh)
    after_rows = df.shape[0]
    print(f"  - Dropped {before_rows - after_rows} rows (>30% NaN)")

    # Fill remaining NaNs
    df = df.ffill().bfill()

    # Convert numeric to float32
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    for col in numeric_cols:
        if col != "Date":
            df[col] = df[col].astype("float32")

    return df


def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".parquet")]

    print(f"[INFO] Found {len(files)} company merged datasets to clean.\n")

    for f in files:
        ticker = f.replace("_merged.parquet", "")
        input_path = f"{INPUT_DIR}/{f}"
        output_path = f"{OUTPUT_DIR}/{ticker}_clean.parquet"

        print(f"[INFO] Cleaning → {f}")
        try:
            df = pd.read_parquet(input_path)
            cleaned = clean_company(df)
            cleaned.to_parquet(output_path)
            print(f"[OK] Saved cleaned dataset → {output_path}\n")
        except Exception as e:
            print(f"[ERROR] Failed to clean {f}: {e}\n")

    print("[DONE] All company datasets cleaned.")


if __name__ == "__main__":
    main()
