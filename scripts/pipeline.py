#!/usr/bin/env python3
"""
pipeline.py

Full data pipeline runner for StockPred.

Stages (in order):

1. FETCH
   - fetch_company.py        → data/raw_companies
   - fetch_macro_main.py     → data/raw_macro
   - fetch_macro_market.py   → data/raw_macro_market
   - fetch_macro_news.py     → data/news/raw

2. CLEAN
   - clean_company.py        → cleaned companies (if you keep a cleaned layer)
   - clean_market.py         → cleaned macro/market (if used)
   - clean_macro.py          → cleaned global macro (if used)

   (If your current design cleans at raw-level, adapt these calls accordingly,
    or comment them out.)

3. COMPUTE
   - compute_indicators.py       → data/processed_*
   - compute_calendar_features.py→ data/processed_calendar/calendar_master.parquet
   - compute_news_sentiment.py   → data/news/sentiment

4. MERGE
   - merge_all_features.py       → data/tft_ready/<TICKER>.parquet
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import argparse


# =========================================================
# FETCH
# =========================================================
def run_fetch():
    try:
        from scripts.data.fetch.fetch_company import main as fetch_company_main
        from scripts.data.fetch.fetch_macro_main import main as fetch_macro_main
        from scripts.data.fetch.fetch_macro_market import main as fetch_macro_market_main
        from scripts.data.fetch.fetch_macro_news import main as fetch_macro_news_main
    except ImportError as e:
        print(f"[WARN] FETCH scripts not found: {e}")
        return

    print("\n========== FETCH STAGE ==========\n")
    fetch_company_main()
    fetch_macro_main()
    fetch_macro_market_main()
    fetch_macro_news_main()


# =========================================================
# CLEAN
# =========================================================
def run_clean():
    try:
        from scripts.data.clean.company_clean import main as clean_company_main
        from scripts.data.clean.macro_clean import main as clean_macro_main
        from scripts.data.clean.market_clean import main as clean_market_main
    except ImportError as e:
        print(f"[WARN] CLEAN scripts not found: {e}")
        return

    print("\n========== CLEAN STAGE ==========\n")
    clean_company_main()
    clean_macro_main()
    clean_market_main()


# =========================================================
# COMPUTE
# =========================================================
def run_compute():
    try:
        from scripts.data.compute.compute_indicators import main as indicators_main
        from scripts.data.compute.compute_calendar_features import main as calendar_main
        from scripts.data.compute.compute_news_sentiment import main as news_sentiment_main
    except ImportError as e:
        print(f"[WARN] COMPUTE scripts not found: {e}")
        return

    print("\n========== COMPUTE STAGE ==========\n")
    indicators_main()
    calendar_main()
    news_sentiment_main()


# =========================================================
# MERGE
# =========================================================
def run_merge():
    try:
        from scripts.data.merge.merge_all_data import main as merge_main
    except ImportError as e:
        print(f"[WARN] MERGE script not found: {e}")
        return

    print("\n========== MERGE STAGE ==========\n")
    merge_main()


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Full StockPred data pipeline")
    parser.add_argument("--no-fetch", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--no-compute", action="store_true")
    parser.add_argument("--no-merge", action="store_true")

    args = parser.parse_args()

    if not args.no_fetch:
        run_fetch()
    else:
        print("[SKIP] FETCH stage skipped.")

    if not args.no_clean:
        run_clean()
    else:
        print("[SKIP] CLEAN stage skipped.")

    if not args.no_compute:
        run_compute()
    else:
        print("[SKIP] COMPUTE stage skipped.")

    if not args.no_merge:
        run_merge()
    else:
        print("[SKIP] MERGE stage skipped.")

    print("\n[PIPELINE COMPLETE]\n")


if __name__ == "__main__":
    main()
