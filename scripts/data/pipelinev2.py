#!/usr/bin/env python3
"""
pipeline.py

Unified macro data pipeline for StockPred.

Runs:
  - ABS GDP SDMX
  - ABS CPI
  - ABS Labour Force
  - ABS Terms of Trade
  - ABS Wage Price Index
  - ABS House Price Index
  - RBA F1/F2 macro (cash rate, yields)
  - RBA F3 (corporate A/BBB yields + credit spreads)
  - FRED macro

Usage:
  python3 -m scripts.pipeline
  python3 -m scripts.pipeline --only abs
  python3 -m scripts.pipeline --only rba
  python3 -m scripts.pipeline --only fred
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Pretty printing utilities
# ---------------------------------------------------------------------
def banner(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def run(script, desc):
    banner(f"[RUN] {desc}")
    try:
        subprocess.run([sys.executable, "-m", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed: {script}")
        print(e)
    print()


# ---------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------
ABS_SCRIPTS = [
    ("scripts.data.fetch.fetch_abs_gdp_sdmx", "ABS GDP (SDMX XML)"),
    ("scripts.data.fetch.fetch_abs_cpi", "ABS CPI (Inflation)"),
    ("scripts.data.fetch.fetch_abs_labour", "ABS Labour Force / Unemployment"),
    ("scripts.data.fetch.fetch_abs_terms_of_trade", "ABS Terms of Trade"),
    ("scripts.data.fetch.fetch_abs_wpi", "ABS Wage Price Index (WPI)"),
    ("scripts.data.fetch.fetch_abs_hpi", "ABS House Price Index (HPI)"),
]

RBA_SCRIPTS = [
    ("scripts.data.fetch.fetch_abs_rba_macro", "RBA F1/F2 (Cash Rate + Gov Bond Yields)"),
    ("scripts.data.fetch.fetch_rba_f3_credit", "RBA F3 (Corporate Yields + Credit Spreads)"),
]

FRED_SCRIPTS = [
    ("scripts.data.fetch.fetch_fred_macro", "US FRED Macro"),
]


# ---------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["abs", "rba", "fred"], help="Run only one part of the pipeline")
    args = parser.parse_args()

    banner("STOCKPRED MACRO DATA PIPELINE")

    # --- ABS ----------------------------------------------------------
    if args.only is None or args.only == "abs":
        banner("[ABS] AUSTRALIAN BUREAU OF STATISTICS MACRO FETCH")
        for module, desc in ABS_SCRIPTS:
            run(module, desc)

    # --- RBA ----------------------------------------------------------
    if args.only is None or args.only == "rba":
        banner("[RBA] RESERVE BANK OF AUSTRALIA MACRO FETCH")
        for module, desc in RBA_SCRIPTS:
            run(module, desc)

    # --- FRED ---------------------------------------------------------
    if args.only is None or args.only == "fred":
        banner("[FRED] US MACROECONOMIC SERIES")
        for module, desc in FRED_SCRIPTS:
            run(module, desc)

    banner("[COMPLETE] FULL MACRO PIPELINE FINISHED")


if __name__ == "__main__":
    main()

