#!/usr/bin/env python3
"""
fetch_rba_f3_credit.py  (patched for encoding issues)

Extract credit spreads from RBA F3 corporate bond data:
    data/raw_rba/f3-data.csv

Handles:
    - Windows-1252 encoding
    - Smart dashes, weird characters
    - Multi-row header garbage

Outputs:
    AUS_CREDIT_A.parquet
    AUS_CREDIT_BBB.parquet
    AUS_CREDIT_SPREAD_A.parquet
    AUS_CREDIT_SPREAD_BBB.parquet
"""

import os
import pandas as pd
import numpy as np

F3_PATH = "data/raw_rba/f3-data.csv"
OUT_DIR = "data/raw_rba"

# Optional extra yield files
F2_10Y = "data/raw_rba/AUS_RBA_YIELD_10Y.parquet"


def clean_col(col):
    if not isinstance(col, str):
        return col
    return (
        col.replace("\x96", "-")
           .replace("\u2013", "-")  # en-dash
           .replace("\u2014", "-")  # em-dash
           .strip()
    )


def detect_header_row(raw_lines):
    """
    F3 CSV often has:
        - metadata rows
        - title rows
        - description rows
    The REAL header row begins with: 'Series ID'
    """
    for i, line in enumerate(raw_lines):
        if "Series ID" in line:
            return i
    return 0  # fallback


def load_gov_10y():
    if not os.path.exists(F2_10Y):
        print("[WARN] gov 10Y not found. Credit spreads will be empty.")
        return None
    df = pd.read_parquet(F2_10Y)
    df = df.rename(columns={"value": "gov_10Y"})
    return df[["date", "gov_10Y"]]


def main():
    print("\n[INFO] Loading RBA F3 corporate data…")

    if not os.path.exists(F3_PATH):
        raise FileNotFoundError(f"Missing {F3_PATH}")

    # ----------------------------------------------------
    # STEP 1 — Read raw bytes & detect header row
    # ----------------------------------------------------
    with open(F3_PATH, "rb") as f:
        raw = f.read()

    # decode with latin1 to avoid Unicode errors
    text = raw.decode("latin1")

    lines = text.splitlines()
    header_idx = detect_header_row(lines)

    print(f"[INFO] Detected header row at line {header_idx}")

    df = pd.read_csv(
        F3_PATH,
        encoding="latin1",
        skiprows=header_idx,
    )

    # sanitize column names
    df.columns = [clean_col(c) for c in df.columns]

    # Ensure first column is date
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.rename(columns={date_col: "date"})

    # ----------------------------------------------------
    # STEP 2 — Extract A-rated and BBB-rated bond yields
    # ----------------------------------------------------
    a_cols = [c for c in df.columns if "FNFYA" in c]
    bbb_cols = [c for c in df.columns if "FNFYBBB" in c]

    print(f"[INFO] A-rated yield columns found: {len(a_cols)}")
    print(f"[INFO] BBB-rated yield columns found: {len(bbb_cols)}")

    df_a = df[["date"] + a_cols].melt("date", var_name="series", value_name="value")
    df_bbb = df[["date"] + bbb_cols].melt("date", var_name="series", value_name="value")

    # Clean names
    df_a["series"] = df_a["series"].str.replace("FNFYA", "A_Yield_")
    df_bbb["series"] = df_bbb["series"].str.replace("FNFYBBB", "BBB_Yield_")

    # numeric
    df_a["value"] = pd.to_numeric(df_a["value"], errors="coerce")
    df_bbb["value"] = pd.to_numeric(df_bbb["value"], errors="coerce")

    # ----------------------------------------------------
    # STEP 3 — Save raw yields
    # ----------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)

    df_a["region"] = "AUS"
    df_a["source"] = "RBA_F3"
    df_a.to_parquet(os.path.join(OUT_DIR, "AUS_CREDIT_A.parquet"), index=False)

    df_bbb["region"] = "AUS"
    df_bbb["source"] = "RBA_F3"
    df_bbb.to_parquet(os.path.join(OUT_DIR, "AUS_CREDIT_BBB.parquet"), index=False)

    print("[OK] Saved raw A and BBB yields")

    # ----------------------------------------------------
    # STEP 4 — Compute credit spreads
    # ----------------------------------------------------
    gov10 = load_gov_10y()

    if gov10 is None:
        print("[WARN] Missing gov 10Y. Skipping credit spread computation.")
        return

    spreads_a = []
    spreads_bbb = []

    for tenor in ["3M", "5M", "7M", "10M"]:
        col_a = f"A_Yield_{tenor}"
        col_bbb = f"BBB_Yield_{tenor}"

        # A spread
        da = df_a[df_a["series"] == col_a].merge(gov10, on="date", how="left")
        if len(da) > 0:
            da["spread"] = da["value"] - da["gov_10Y"]
            da["series"] = f"A_Spread_{tenor}"
            spreads_a.append(da[["date", "spread", "series"]])

        # BBB spread
        dbb = df_bbb[df_bbb["series"] == col_bbb].merge(gov10, on="date", how="left")
        if len(dbb) > 0:
            dbb["spread"] = dbb["value"] - dbb["gov_10Y"]
            dbb["series"] = f"BBB_Spread_{tenor}"
            spreads_bbb.append(dbb[["date", "spread", "series"]])

    if spreads_a:
        out = pd.concat(spreads_a)
        out["region"] = "AUS"
        out["source"] = "RBA_F3"
        out.to_parquet(os.path.join(OUT_DIR, "AUS_CREDIT_SPREAD_A.parquet"), index=False)
        print("[OK] Saved A-rated credit spreads")

    if spreads_bbb:
        out = pd.concat(spreads_bbb)
        out["region"] = "AUS"
        out["source"] = "RBA_F3"
        out.to_parquet(os.path.join(OUT_DIR, "AUS_CREDIT_SPREAD_BBB.parquet"), index=False)
        print("[OK] Saved BBB-rated credit spreads")

    print("\n[COMPLETE] RBA F3 credit extraction finished.\n")


if __name__ == "__main__":
    main()
