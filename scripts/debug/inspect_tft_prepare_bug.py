#!/usr/bin/env python3
"""
inspect_tft_prepare_bug.py

This script identifies why sector_id becomes 'Unknown' during
prepare_multiseries_tft_dataset.py.

It inspects:

  - Actual filenames in data/tft_ready/
  - Expected sector keys in data.yaml
  - Normalization (underscore ↔ dot)
  - Which tickers fail mapping
"""

import os
import yaml

TFT_READY_DIR = "data/tft_ready"
YAML_PATH = "config/data.yaml"


def load_sector_map():
    with open(YAML_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["companies"]["tickers_with_sectors"]


def normalize_filename_to_ticker(fname):
    """
    Reproduce the logic from prepare script.
    """
    ticker = fname.replace(".parquet", "")
    return ticker  # BEFORE applying fix


def normalize_with_fix(fname):
    """
    FIXED version (underscore → dot).
    """
    ticker = fname.replace(".parquet", "")
    ticker = ticker.replace("_", ".")
    return ticker


def main():
    print("======================================================")
    print("  INSPECTING TFT PREPARE BUG — SECTOR MAPPING CHECK")
    print("======================================================\n")

    if not os.path.exists(TFT_READY_DIR):
        print(f"[ERROR] Directory missing: {TFT_READY_DIR}")
        return

    sector_map = load_sector_map()
    print(f"[INFO] Loaded {len(sector_map)} sector mappings from YAML.\n")

    files = sorted([f for f in os.listdir(TFT_READY_DIR) if f.endswith(".parquet")])

    if not files:
        print("[ERROR] No parquet files found in data/tft_ready")
        return

    print("[INFO] Found merged files:")
    for f in files:
        print("  -", f)

    print("\n==============================")
    print("CHECKING FILE → TICKER MATCH")
    print("==============================\n")

    unknown_before = []
    unknown_after = []

    for f in files:
        raw = normalize_filename_to_ticker(f)
        fixed = normalize_with_fix(f)

        has_raw = raw in sector_map
        has_fixed = fixed in sector_map

        print(f"FILE: {f}")
        print(f"  raw ticker   = {raw:15s} | in YAML: {has_raw}")
        print(f"  fixed ticker = {fixed:15s} | in YAML: {has_fixed}")

        if not has_raw:
            unknown_before.append(raw)
        if not has_fixed:
            unknown_after.append(fixed)

        print()

    print("\n==============================")
    print(" SUMMARY ")
    print("==============================\n")

    print("Tickers UNKNOWN BEFORE FIX (current script):")
    for u in unknown_before:
        print("  ❌", u)

    print("\nTickers UNKNOWN AFTER FIX (underscore → dot):")
    for u in unknown_after:
        print("  ⚠️", u)

    print("\nIf UNKNOWN BEFORE is large and UNKNOWN AFTER is 0,")
    print(" → this confirms the bug is filename normalization.")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
