#!/usr/bin/env python3
"""
fetch_abs_rba_macro.py

CURRENT FOCUS:
- Robustly fetch & parse RBA macro series from CSV:
    * RBA_CASH_RATE   (Table F1, column FIRMMCRTD)
    * RBA_YIELD_10Y   (Table F2, column FCMYGBAG10D)
    * RBA_YIELD_2Y    (Table F2, column FCMYGBAG2D)

ABS (CPI / UNEMP / GDP) via SDMX Data API is left as a TODO for now:
- The API responds but the SDMX → tabular mapping is non-trivial.
- You already have US macro via FRED and global markets via Yahoo,
  so we can move forward without AUS ABS for regime IDs.

Outputs:
    data/raw_rba/AUS_RBA_CASH_RATE.parquet
    data/raw_rba/AUS_RBA_YIELD_10Y.parquet
    data/raw_rba/AUS_RBA_YIELD_2Y.parquet

Config fields used (config/data.yaml):

australia_macro:
  canonical_prefix: "AUS"
  sources:
    RBA_CASH_RATE: "f1-data"
    RBA_YIELD_10Y: "f2-data"
    RBA_YIELD_2Y:  "f2-data"

data:
  sources:
    rba_folder: "data/raw_rba"
  processed:
    rba_folder: "data/processed_rba"
"""

import os
import io
from typing import Optional, Tuple, Dict

import yaml
import pandas as pd
import requests

CONFIG_PATH = "config/data.yaml"


# ---------------------------------------------------------------------
# CONFIG LOADER
# ---------------------------------------------------------------------
def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# RBA HELPERS
# ---------------------------------------------------------------------
def download_rba_table(table_id: str) -> Optional[str]:
    """
    Download raw RBA CSV text for a given table id, e.g. 'f1-data', 'f2-data'.

    Returns the CSV text (string) or None on failure.
    """
    url = f"https://www.rba.gov.au/statistics/tables/csv/{table_id}.csv"
    print(f"[INFO] RBA request → {url}")
    try:
        r = requests.get(url, timeout=30)
    except Exception as e:
        print(f"[ERROR] RBA request failed for {table_id}: {e}")
        return None

    if r.status_code != 200:
        print(f"[WARN] RBA table {table_id} returned HTTP {r.status_code}")
        return None

    return r.text


def parse_rba_csv(text: str) -> pd.DataFrame:
    """
    Parse an RBA CSV (F1/F2) into a DataFrame.

    The RBA CSV format has:
        - a metadata block
        - a line starting with 'Series ID,...'
        - then data rows where first column is the date.

    We:
        1) Find the 'Series ID' header row.
        2) Rebuild CSV from that row onwards.
        3) Parse with pandas, header=0.
    """
    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        # Be generous: match 'Series ID' at start or as first token
        if line.startswith("Series ID") or line.startswith("Series ID,"):
            header_idx = i
            break

    if header_idx is None:
        # As a fallback, try to find the first line containing 'Series ID'
        for i, line in enumerate(lines):
            if "Series ID" in line:
                header_idx = i
                break

    if header_idx is None:
        raise ValueError("Could not locate 'Series ID' header row in RBA CSV")

    data_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(data_text), header=0)

    return df


def tidy_rba_series(
    df: pd.DataFrame,
    value_column: str,
    series_name: str,
    region: str,
) -> pd.DataFrame:
    """
    Given a parsed RBA table DataFrame, extract a single series.

    - Date column is assumed to be 'Series ID' (first column in RBA tables).
    - value_column is one of:
        * FIRMMCRTD  (cash rate target, F1)
        * FCMYGBAG10D (10Y yield, F2)
        * FCMYGBAG2D  (2Y yield, F2)
    """
    # Robust date column detection
    cols = list(df.columns)
    if "Series ID" in cols:
        date_col = "Series ID"
    else:
        # fallback: first column
        date_col = cols[0]

    if value_column not in df.columns:
        print(f"[WARN] Column {value_column} missing in RBA table; columns={cols}")
        return pd.DataFrame()

    sub = df[[date_col, value_column]].copy()
    sub.rename(columns={date_col: "date", value_column: "value"}, inplace=True)

    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.dropna(subset=["date"])

    # Ensure numeric
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["value"])

    sub["series"] = series_name
    sub["region"] = region
    sub["source"] = "RBA"

    # Sort by date
    sub = sub.sort_values("date").reset_index(drop=True)
    return sub


def fetch_rba_series(
    name: str,
    table_id: str,
    region: str,
) -> Optional[pd.DataFrame]:
    """
    High-level wrapper:
    - Download F1/F2 table
    - Parse it
    - Extract correct column based on logical series name
    """
    text = download_rba_table(table_id)
    if text is None:
        return None

    try:
        df_raw = parse_rba_csv(text)
    except Exception as e:
        print(f"[ERROR] Failed to parse RBA table {table_id}: {e}")
        return None

    # Map logical name → column code
    if name == "RBA_CASH_RATE":
        value_col = "FIRMMCRTD"    # Cash Rate Target
    elif name == "RBA_YIELD_10Y":
        value_col = "FCMYGBAG10D"  # 10-year government bond yield
    elif name == "RBA_YIELD_2Y":
        value_col = "FCMYGBAG2D"   # 2-year government bond yield
    else:
        print(f"[WARN] Unknown RBA series logical name: {name}")
        return None

    df = tidy_rba_series(df_raw, value_col, name, region)
    if df.empty:
        print(f"[WARN] RBA series {name} produced empty DataFrame")
        return None

    return df


# ---------------------------------------------------------------------
# ABS HELPERS (STUBBED)
# ---------------------------------------------------------------------
def fetch_abs_series_stub(name: str, sid: str, region: str) -> Optional[pd.DataFrame]:
    """
    Placeholder for future ABS SDMX integration.

    Right now, ABS Data API is non-trivial to flatten cleanly without either:
        - using sdmxabs library, or
        - going through Indicator API (requires API key).

    For now we simply log and return None so the rest of the pipeline can run.
    """
    print(
        f"[INFO] ABS fetch for {name} (series={sid}) is currently disabled. "
        "Skipping and relying on US FRED + RBA macro for regimes."
    )
    return None


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    cfg = load_config()

    aus_cfg = cfg["australia_macro"]
    aus_sources: Dict[str, str] = aus_cfg["sources"]
    region = aus_cfg.get("canonical_prefix", "AUS")

    raw_rba_dir = cfg["data"]["sources"]["rba_folder"]
    os.makedirs(raw_rba_dir, exist_ok=True)

    # -------------------------------
    # 1) ABS (currently stubbed)
    # -------------------------------
    print("\n==============================================")
    print(" ABS MACRO SERIES (CURRENTLY SKIPPED)")
    print("==============================================\n")

    for name, sid in aus_sources.items():
        if name.startswith("ABS_"):
            fetch_abs_series_stub(name, sid, region)

    # -------------------------------
    # 2) RBA series from F1/F2
    # -------------------------------
    print("\n==============================================")
    print(" FETCHING RBA MACRO SERIES (F1, F2 TABLES)")
    print("==============================================\n")

    for name, table_id in aus_sources.items():
        if not name.startswith("RBA_"):
            continue

        print(f"[INFO] Fetching RBA series {name} from table {table_id} ...")
        df = fetch_rba_series(name, table_id, region)
        if df is None or df.empty:
            continue

        # Save raw (cleaned) parquet
        fname = f"{region}_{name}.parquet"
        out_path = os.path.join(raw_rba_dir, fname)
        df.to_parquet(out_path, index=False)
        print(f"[OK] Saved RBA series → {out_path} (rows={len(df)})")

    print("\n[COMPLETE] ABS (skipped) + RBA macro fetch finished.\n")


if __name__ == "__main__":
    main()
