#!/usr/bin/env python3
"""
fetch_macro_news.py

Fetches market news from yfinance for all assets listed in:
    config/data.yaml → news.tickers

Outputs JSON files for compute_news_sentiment.py.

Now with:
- Canonical naming via canonical_map.py
- Ensures news files are named by canonical ID (AXJO, GSPC, GOLD, DXY, ...)
"""

import os
import json
from datetime import datetime

import yaml
import yfinance as yf

from scripts.data.fetch.canonical_map import canonical_name, safe_filename

CONFIG_PATH = "config/data.yaml"


# ---------------------------------------------------------------------
# CONFIG LOADER
# ---------------------------------------------------------------------
def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    cfg = load_config()

    # Example from your data.yaml:
    # news:
    #   tickers:
    #     AXJO: "^AXJO"
    #     SPY: "SPY"
    #     GSPC: "^GSPC"
    #     ...
    news_map = cfg["news"]["tickers"]

    out_dir = cfg["data"]["sources"]["news_folder"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Fetching news for {len(news_map)} assets")

    for alias, symbol in news_map.items():
        # Decide canonical name:
        # 1) Prefer the alias if it's a known canonical
        # 2) Otherwise derive from the symbol
        canon_from_alias = canonical_name(alias)
        canon_from_symbol = canonical_name(symbol)

        # If alias and symbol map to different canonicals,
        # we log it and still prefer the alias, because that's
        # what your config is keyed by.
        if canon_from_alias != canon_from_symbol:
            print(
                f"[WARN] Alias '{alias}' → {canon_from_alias}, "
                f"symbol '{symbol}' → {canon_from_symbol}. "
                f"Using alias canonical='{canon_from_alias}'."
            )

        canonical_id = canon_from_alias
        safe_canon = safe_filename(canonical_id)

        print(f"[INFO] Fetching news for {alias} ({symbol}) → canonical='{canonical_id}'")

        try:
            yf_ticker = yf.Ticker(symbol)
            news = yf_ticker.news  # list[dict]
        except Exception as e:
            print(f"[ERROR] Could not fetch news for {symbol}: {e}")
            continue

        if not news:
            print(f"[WARN] No news for {symbol}")
            continue

        outfile = os.path.join(out_dir, f"{safe_canon}.json")
        with open(outfile, "w") as f:
            json.dump(news, f, indent=2)

        print(f"[OK] Saved news → {outfile}")

    print("\n[COMPLETE] Market news fetch done.\n")


if __name__ == "__main__":
    main()
