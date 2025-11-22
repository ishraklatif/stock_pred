#!/usr/bin/env python3
"""
fetch_macro_news.py

Fetches market news from yfinance for all assets listed in config/data.yaml → news.tickers.
Outputs JSON files for compute_news_sentiment.py.
"""

import os
import json
from datetime import datetime
import yaml
import yfinance as yf

CONFIG_PATH = "config/data.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    news_map = cfg["news"]["tickers"]
    out_dir = cfg["data"]["sources"]["news_folder"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Fetching news for {len(news_map)} assets")

    for name, symbol in news_map.items():
        print(f"[INFO] Fetching news for {name} ({symbol})")

        try:
            yf_ticker = yf.Ticker(symbol)
            news = yf_ticker.news  # list of dicts
        except Exception as e:
            print(f"[ERROR] Could not fetch news for {symbol}: {e}")
            continue

        if not news:
            print(f"[WARN] No news for {symbol}")
            continue

        outfile = os.path.join(out_dir, f"{name}.json")
        with open(outfile, "w") as f:
            json.dump(news, f, indent=2)

        print(f"[OK] Saved → {outfile}")

    print("\n[COMPLETE] Market news fetch done.\n")


if __name__ == "__main__":
    main()
