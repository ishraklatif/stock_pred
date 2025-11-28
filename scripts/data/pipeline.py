#!/usr/bin/env python3
"""
Unified StockPred Pipeline

This script runs ALL components of your data infrastructure with
modular flags. Every stage is individually triggerable.

STAGES AND FLAGS
----------------
ABS fetch              : --abs
RBA fetch              : --rba
FRED fetch             : --fred

Company OHLCV          : --company
Global macro indices   : --macro
Macro-market assets    : --market
News fetch             : --news

CLEAN stage            : --clean
COMPUTE stage          : --compute
MERGE stage            : --merge

Run everything         : --all

EXAMPLES
--------
# Run everything
python3 -m scripts.pipeline --all

# Only fetch RBA + FRED
python3 -m scripts.pipeline --rba --fred

# Full data build without refetching
python3 -m scripts.pipeline --clean --compute --merge
"""

import argparse
import subprocess
import sys
from pathlib import Path



# ---------------------------------------------------------------------
# Pretty banners
# ---------------------------------------------------------------------
def banner(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80 + "\n")


def run(module, desc):
    banner(f"[RUN] {desc}")
    try:
        subprocess.run([sys.executable, "-m", module], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed: {module}")
        print(e)
    print()


# ---------------------------------------------------------------------
# Script groups
# ---------------------------------------------------------------------
ABS_SCRIPTS = [
    ("scripts.data.fetch.fetch_abs_gdp_sdmx", "ABS GDP (SDMX XML)"),
    ("scripts.data.fetch.fetch_abs_cpi", "ABS CPI (Inflation)"),
    ("scripts.data.fetch.fetch_abs_labour", "ABS Labour Force"),
    ("scripts.data.fetch.fetch_abs_terms_of_trade", "ABS Terms of Trade"),
    ("scripts.data.fetch.fetch_abs_wpi", "ABS Wage Price Index"),
    ("scripts.data.fetch.fetch_abs_hpi", "ABS House Price Index"),
]

RBA_SCRIPTS = [
    ("scripts.data.fetch.fetch_abs_rba_macro", "RBA Cash Rate + Gov Yields (F1/F2)"),
    ("scripts.data.fetch.fetch_rba_f3_credit", "RBA Corporate Yields + Credit Spreads (F3)"),
]

FRED_SCRIPTS = [
    ("scripts.data.fetch.fetch_fred_macro", "US FRED Macro Series"),
]

COMPANY_SCRIPTS = [
    ("scripts.data.fetch.fetch_company", "Company OHLCV (YFinance)"),
]

MACRO_SCRIPTS = [
    ("scripts.data.fetch.fetch_macro_main", "Global Macro Indices (AXJO, GSPC, FTSE, N225, etc.)"),
]

MARKET_SCRIPTS = [
    ("scripts.data.fetch.fetch_macro_market", "Macro-Market Assets (DXY, GOLD, OIL, FX, Commodities)"),
]

MACRO_EXTRA_SCRIPTS = [
    ("scripts.data.fetch.fetch_move_index", "MOVE Index (FRED)"),
    ("scripts.data.fetch.fetch_vvix", "VVIX Index (^VVIX)"),
    ("scripts.data.fetch.fetch_us_credit_spreads", "US Credit Spreads (HY, IG, TED, LIBOR-OIS)"),
    ("scripts.data.fetch.fetch_us_yield_curve_extra", "US Yield Curve Extra (3M, 5Y, 30Y)"),
]


NEWS_SCRIPTS = [
    ("scripts.data.fetch.fetch_macro_news", "Market News"),
]

CLEAN_SCRIPTS = [
    ("scripts.data.clean.company_clean", "Clean Companies"),
    ("scripts.data.clean.macro_clean", "Clean Global Macro"),
    ("scripts.data.clean.market_clean", "Clean Macro-Market"),
    ("scripts.data.clean.clean_abs", "Clean ABS"),

    # Extract ABS time-series into abs_all_series/
    ("scripts.data.extract.extract_abs_series", "Extract ALL ABS Time-Series (GDP, ToT, subseries)"),

    # FINAL filtered PCA GDP/ToT extraction
    ("scripts.data.extract.extract_abs_gdp_tot_filtered_v3", "Compute PCA GDP + PCA ToT (Filtered Option C)"),

    ("scripts.data.clean.clean_rba", "Clean RBA"),
    ("scripts.data.clean.clean_fred", "Clean FRED"),
    ("scripts.data.clean.clean_macro_extra", "Clean Macro-Extra"),
    ("scripts.data.clean.clean_sector", "Clean Sector ETFs"),
]






COMPUTE_SCRIPTS = [
    ("scripts.data.compute.compute_indicators", "Indicators + Technical Features"),
    ("scripts.data.compute.compute_calendar_features", "Calendar Features"),
    ("scripts.data.compute.compute_news_sentiment", "News Sentiment"),
]

MERGE_SCRIPTS = [
    ("scripts.data.merge.merge_all_features", "Merge Into TFT/PatchTST Ready Dataset"),
]


# ---------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified StockPred Data Pipeline")

    parser.add_argument("--all", action="store_true", help="Run everything")

    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--rba", action="store_true")
    parser.add_argument("--fred", action="store_true")

    parser.add_argument("--company", action="store_true")
    parser.add_argument("--macro", action="store_true")
    parser.add_argument("--market", action="store_true")
    parser.add_argument("--macroextra", action="store_true", help="Fetch MOVE/VVIX/spreads/yields")

    parser.add_argument("--news", action="store_true")

    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--merge", action="store_true")

    args = parser.parse_args()

    # If --all is used → activate all flags
    if args.all:
        args.abs = args.rba = args.fred = True
        args.company = args.macro = args.market = args.news = True
        args.clean = args.compute = args.merge = True
        args.macroextra = True

    banner("STOCKPRED — UNIFIED DATA PIPELINE")

    # ------------------------------
    # FETCH BLOCKS
    # ------------------------------
    if args.abs:
        banner("[ABS] Fetching ABS Macro Series")
        for m, desc in ABS_SCRIPTS:
            run(m, desc)

    if args.rba:
        banner("[RBA] Fetching RBA Macro Series")
        for m, desc in RBA_SCRIPTS:
            run(m, desc)

    if args.fred:
        banner("[FRED] Fetching US FRED Macro")
        for m, desc in FRED_SCRIPTS:
            run(m, desc)

    if args.company:
        banner("[COMPANIES] Fetching OHLCV")
        for m, desc in COMPANY_SCRIPTS:
            run(m, desc)

    if args.macro:
        banner("[MACRO] Fetching Global Macro Indices")
        for m, desc in MACRO_SCRIPTS:
            run(m, desc)

    if args.market:
        banner("[MARKET] Fetching Macro-Market Assets")
        for m, desc in MARKET_SCRIPTS:
            run(m, desc)
    
    if args.macroextra:
        banner("[MACRO-EXTRA] Fetching Additional Macro Series")
        for m, desc in MACRO_EXTRA_SCRIPTS:
            run(m, desc)


    if args.news:
        banner("[NEWS] Fetching Market News")
        for m, desc in NEWS_SCRIPTS:
            run(m, desc)

    # ------------------------------
    # CLEAN
    # ------------------------------
    if args.clean:
        banner("[CLEAN] Cleaning All Data")
        for m, desc in CLEAN_SCRIPTS:
            run(m, desc)

    # ------------------------------
    # COMPUTE
    # ------------------------------
    if args.compute:
        banner("[COMPUTE] Computing Features")
        for m, desc in COMPUTE_SCRIPTS:
            run(m, desc)

    # ------------------------------
    # MERGE
    # ------------------------------
    if args.merge:
        banner("[MERGE] Merging Final TFT/PatchTST Dataset")
        for m, desc in MERGE_SCRIPTS:
            run(m, desc)

    banner("[COMPLETE] PIPELINE FINISHED")


if __name__ == "__main__":
    main()


