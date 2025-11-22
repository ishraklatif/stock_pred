#!/usr/bin/env python3

import yaml
import yfinance as yf
from datetime import datetime, timedelta

CONFIG = "config/data.yaml"

def quick_check(ticker):
    try:
        df = yf.download(ticker, period="5d", progress=False)
        return not df.empty
    except:
        return False

cfg = yaml.safe_load(open(CONFIG))

tickers = list(cfg["companies"]["tickers_with_sectors"].keys())

valid = []
invalid = []

for t in tickers:
    print(f"Checking {t}...")
    if quick_check(t):
        valid.append(t)
    else:
        invalid.append(t)

print("\n=== VALID TICKERS ===")
print(valid)

print("\n=== INVALID / DELISTED ===")
print(invalid)
