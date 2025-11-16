#!/usr/bin/env python3
"""
fetch_news_headlines.py

Fetches news headlines for major markets, commodities, and FX pairs.
Saves raw JSON files in data/news/raw/.
Uses yfinance news API (free and reliable).
"""

import os
import json
import yfinance as yf

OUTPUT_DIR = "data/level4/raw_news"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tickers for news
TICKERS = {
    "AXJO": "^AXJO",
    "SPY": "SPY",
    "GSPC": "^GSPC",
    "N225": "^N225",
    "HSI": "^HSI",
    "SSE": "000001.SS",
    "CSI300": "000300.SS",

    "GOLD": "GC=F",
    "OIL": "BZ=F",
    "COPPER": "HG=F",

    "AUDUSD": "AUDUSD=X",
    "AUDJPY": "AUDJPY=X",
    "DXY": "DX-Y.NYB",
}

def main():
    print("[INFO] Fetching news from yfinance...")

    for name, ticker in TICKERS.items():
        print(f"[INFO] Fetching news for {name} ({ticker})")

        try:
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
        except Exception as e:
            print(f"[ERROR] Failed to fetch news for {name}: {e}")
            continue

        out_path = f"{OUTPUT_DIR}/{name}_news.json"
        with open(out_path, "w") as f:
            json.dump(news, f, indent=2)

        print(f"[OK] Saved → {out_path}")

    print("\n[DONE] Raw news downloads complete.\n")

if __name__ == "__main__":
    main()

# python3 scripts/fetch_news_headlines.py
