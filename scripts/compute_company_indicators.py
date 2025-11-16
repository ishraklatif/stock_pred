#!/usr/bin/env python3
"""
compute_company_indicators.py

Computes technical indicators for company OHLCV files.
Uses lowercase column names: date, open, high, low, close, adj_close, volume
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = "data/raw_companies"
OUT_DIR = "data/processed_companies"
os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------
# Indicator functions (pure pandas)
# -----------------------------------

def ema(close, span):
    return close.ewm(span=span, adjust=False).mean()


def rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(close):
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def bollinger(close, n=20, std=2):
    ma = close.rolling(n).mean()
    sigma = close.rolling(n).std()
    return ma + std * sigma, ma, ma - std * sigma


def atr(high, low, close, n=14):
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# -----------------------------------
# Indicator computation
# -----------------------------------

def compute_indicators(df):
    df = df.copy()

    # Required columns
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise RuntimeError(f"Missing column '{col}' in company dataset")

    # Returns
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Moving Averages
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_30"] = df["close"].rolling(30).mean()
    df["ema_10"] = ema(df["close"], 10)
    df["ema_30"] = ema(df["close"], 30)

    # RSI
    df["rsi_14"] = rsi(df["close"])

    # MACD
    macd_line, signal, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    # Bollinger Bands
    upper, middle, lower = bollinger(df["close"])
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower

    # ATR
    df["atr_14"] = atr(df["high"], df["low"], df["close"])

    # Volatility features
    df["vol_7"] = df["return"].rolling(7).std()
    df["vol_30"] = df["return"].rolling(30).std()
    df["vol_90"] = df["return"].rolling(90).std()

    return df


# -----------------------------------
# Main execution
# -----------------------------------

def main():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".parquet")]
    print(f"[INFO] Found {len(files)} raw company files.")

    for file in files:
        ticker = file.replace(".parquet", "")
        print(f"[INFO] Computing indicators for: {ticker}")

        df = pd.read_parquet(f"{RAW_DIR}/{file}")

        # ensure lowercase columns (in case the raw file had unexpected structure)
        df.columns = [str(c).lower() for c in df.columns]

        if "date" not in df.columns:
            raise RuntimeError(
                f"❌ No 'date' column found in file: {file}. Columns={df.columns.tolist()}"
            )

        df = compute_indicators(df)
        df.to_parquet(f"{OUT_DIR}/{ticker}.parquet", index=False)

        print(f"[OK] Saved indicators for {ticker}")

    print("[COMPLETE] Processed all company indicators.")


if __name__ == "__main__":
    main()

# python3 scripts/compute_company_indicators.py
