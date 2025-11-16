#!/usr/bin/env python3
"""
sanitize_columns.py

Sanitize column names for ALL datasets before TFT training.

Supports:
    --macro       → sanitize macro_clean.parquet
    --companies   → sanitize all company_clean/*.parquet
    --all         → sanitize macro + companies

Output:
    data/final/macro_clean_sanitized.parquet
    data/company_clean_sanitized/<TICKER>_clean_sanitized.parquet

Rules:
    - Replace invalid chars: . - = / space ( and )
    - Collapse multiple underscores
    - Ensure names start with [A-Za-z_]
    - TFT requires strict column safety
"""

import os
import argparse
import pandas as pd
import re


# -----------------------------
# Column sanitisation logic
# -----------------------------
def sanitize(name: str) -> str:
    # Replace invalid characters
    name = re.sub(r"[.\-=/() ]", "_", name)

    # Collapse consecutive underscores
    name = re.sub(r"__+", "_", name)

    # Remove trailing underscores
    name = re.sub(r"_$", "", name)

    # Ensure valid start char
    if not re.match(r"^[A-Za-z_]", name):
        name = "_" + name

    return name


def sanitize_file(input_path: str, output_path: str):
    print(f"[INFO] Sanitizing → {input_path}")
    df = pd.read_parquet(input_path)

    new_cols = {col: sanitize(col) for col in df.columns}
    df = df.rename(columns=new_cols)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)

    print(f"[OK] Saved sanitized → {output_path}")
    print(f"[OK] Total columns: {len(df.columns)}\n")


# -----------------------------
# MAIN EXECUTION
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--macro", action="store_true", help="Sanitize macro_clean.parquet")
    parser.add_argument("--companies", action="store_true", help="Sanitize company_clean datasets")
    parser.add_argument("--all", action="store_true", help="Sanitize both macro + companies")
    args = parser.parse_args()

    # -------- MACRO --------
    if args.macro or args.all:
        macro_in = "data/level4_final/market/macro_clean.parquet"
        macro_out = "data/sanitised_final/market/macro_clean_sanitized.parquet"
        sanitize_file(macro_in, macro_out)

    # -------- COMPANIES --------
    if args.companies or args.all:
        in_dir = "data/level4_final/company"
        out_dir = "data/sanitised_final/company"
        os.makedirs(out_dir, exist_ok=True)

        files = [f for f in os.listdir(in_dir) if f.endswith(".parquet")]
        if not files:
            print("[WARN] No company_clean files found.")
        else:
            print(f"[INFO] Sanitizing {len(files)} company datasets...\n")
            for f in files:
                ticker = f.replace("_clean.parquet", "")
                input_path = f"{in_dir}/{f}"
                output_path = f"{out_dir}/{ticker}_clean_sanitized.parquet"
                sanitize_file(input_path, output_path)

    print("[DONE] Column sanitisation completed.")


if __name__ == "__main__":
    main()
