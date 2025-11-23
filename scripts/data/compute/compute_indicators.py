#!/usr/bin/env python3
"""
compute_indicators.py

Minimal technical indicator computation.
Compatible with cleaned OHLCV + canonical mapping.
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

CONFIG_PATH = "config/data.yaml"
np.seterr(invalid="ignore", divide="ignore")


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
# Indicator helpers
# ----------------------------------------------------------------------
def rsi(close, n=14):
    delta = close.diff()
    up = delta.where(delta > 0, 0).rolling(n).mean()
    down = (-delta.where(delta < 0, 0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bb_percent_b(close, window=20, num_std=2):
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return (close - lower) / (upper - lower)


def candle_body_ratio(df):
    body = df["close"] - df["open"]
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return body / rng


def obv(close, volume):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).fillna(0).cumsum()


# ----------------------------------------------------------------------
# Compute minimal indicators
# ----------------------------------------------------------------------
def compute(df, cfg):
    df = df.copy()

    # Log return
    df["log_return"] = np.log(
        df["close"].replace(0, np.nan)
        / df["close"].shift(1).replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    # BB %b
    bb = cfg["bollinger"]
    df["bb_percent_b"] = bb_percent_b(
        df["close"], window=bb["window"], num_std=bb["num_std"]
    )

    # RSI
    df["rsi_14"] = rsi(df["close"], 14)

    # Candle body ratio
    df["candle_body_ratio"] = candle_body_ratio(df)

    # OBV
    df["obv"] = obv(df["close"], df["volume"]) if "volume" in df else np.nan

    keep = [
        "Date", "open", "high", "low", "close", "volume",
        "log_return", "bb_percent_b", "rsi_14", "candle_body_ratio", "obv"
    ]
    return df[[c for c in keep if c in df.columns]]


# ----------------------------------------------------------------------
# Directory processor
# ----------------------------------------------------------------------
def process(raw_dir, out_dir, ind_cfg):
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for f in raw.glob("*.parquet"):
        print(f"[INFO] Indicators: {f.name}")
        df = pd.read_parquet(f)

        # Normalize & sort
        df.columns = [c.capitalize() if c == "date" else c for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        df = compute(df, ind_cfg)

        df.to_parquet(out / f.name, index=False)
        print(f"[OK] Saved → {out / f.name}")


def main():
    cfg = load_config()
    src = cfg["data"]["sources"]
    dst = cfg["data"]["processed"]
    ind_cfg = cfg["indicators"]

    tasks = [
        (src["companies_folder"], dst["companies_folder"]),
        (src["macro_folder"], dst["macro_folder"]),
        (src["market_folder"], dst["market_folder"]),
    ]

    for raw, out in tasks:
        print(f"\n==== Processing {raw} → {out}")
        process(raw, out, ind_cfg)

    print("\n[COMPLETE] Indicators computed.")


if __name__ == "__main__":
    main()
