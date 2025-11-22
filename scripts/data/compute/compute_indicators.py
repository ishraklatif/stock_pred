#!/usr/bin/env python3
"""
compute_indicators.py

Unified indicator engine for:
- Company OHLCV series
- Macro index series
- Macro-market series (VIX, DXY, FX, commodities, etc.)

Assumes all *raw* parquet files use canonical yfinance-style columns:
    date, open, high, low, close, adj_close (optional), volume (optional)

Reads from config/data.yaml:

data:
  sources:
    companies_folder: "data/raw_companies"
    macro_folder: "data/raw_macro"
    market_folder: "data/raw_macro_market"

  processed:
    companies_folder: "data/processed_companies"
    macro_folder: "data/processed_macro"
    market_folder: "data/processed_macro_market"

indicators:
  sma_windows: [...]
  ema_windows: [...]
  rsi_periods: [...]
  macd: {fast, slow, signal}
"""

import os
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

import numpy as np
np.seterr(invalid='ignore', divide='ignore')


CONFIG_PATH = "config/data.yaml"


# ======================================================================
# CONFIG
# ======================================================================
def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ======================================================================
# INDICATOR FUNCTIONS (expect lowercase: open, high, low, close, volume)
# ======================================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_features(close: pd.Series, n: int = 20, num_std: int = 2):
    ma = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / ma.replace(0, np.nan)
    percent_b = (close - lower) / (upper - lower)
    return upper, ma, lower, width, percent_b


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def wma(series: pd.Series, window: int):
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True,
    )


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_series = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr_series
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr_series

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx_val, plus_di, minus_di


def stochastic_oscillator(high, low, close, k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return k, d


def williams_r(high, low, close, period: int = 14):
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)


def cci(high, low, close, period: int = 20):
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = (tp - sma_tp).abs().rolling(period).mean()
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def parkinson_vol(high, low, window: int = 20):
    hl_ratio = (high / low).replace(0, np.nan)
    return np.sqrt(
        (1 / (4 * np.log(2)))
        * (np.log(hl_ratio) ** 2).rolling(window).mean()
    )


def garman_klass_vol(open_, high, low, close, window: int = 20):
    log_hl = np.log(high / low).replace(0, np.nan)
    log_co = np.log(close / open_).replace(0, np.nan)
    term1 = 0.5 * (log_hl ** 2)
    term2 = (2 * np.log(2) - 1) * (log_co ** 2)
    return np.sqrt((term1 - term2).rolling(window).mean())


def obv(close: pd.Series, volume: pd.Series):
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).fillna(0).cumsum()


def vpt(close: pd.Series, volume: pd.Series):
    pct_change = close.pct_change().fillna(0)
    return (pct_change * volume).cumsum()


def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]

    body = close - open_
    range_ = (high - low).replace(0, np.nan)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    df["candle_body"] = body
    df["candle_range"] = high - low
    df["candle_body_ratio"] = body / range_
    df["candle_upper_wick_ratio"] = upper_wick / range_
    df["candle_lower_wick_ratio"] = lower_wick / range_

    return df


def realized_vol(close: pd.Series, windows=None):
    if windows is None:
        windows = [5, 20, 60, 120]
    logret = np.log(close / close.shift())
    return {f"vol_{w}": logret.rolling(w).std() for w in windows}


def multi_returns(close: pd.Series, windows=None):
    if windows is None:
        windows = [1, 5, 10, 20]
    return {f"ret_{w}": close.pct_change(w) for w in windows}


# ======================================================================
# CORE ENGINE
# ======================================================================
def compute_indicators_for_frame(df: pd.DataFrame, ind_cfg: dict) -> pd.DataFrame:
    """
    Apply indicators to one OHLCV frame.

    Expect columns (lowercase): date, close (+ open/high/low/volume if available).
    """
    df = df.copy()

    if "close" not in df.columns:
        print("  [WARN] Missing 'close' column, skipping indicators.")
        return df

    close = df["close"]
    open_ = df["open"] if "open" in df.columns else None
    high = df["high"] if "high" in df.columns else None
    low = df["low"] if "low" in df.columns else None
    volume = df["volume"] if "volume" in df.columns else None

    sma_windows = ind_cfg.get("sma_windows", [5, 10, 20, 50, 100, 200])
    ema_windows = ind_cfg.get("ema_windows", [5, 10, 20, 50, 100, 200])
    rsi_periods = ind_cfg.get("rsi_periods", [7, 14])
    macd_cfg = ind_cfg.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    macd_fast = macd_cfg.get("fast", 12)
    macd_slow = macd_cfg.get("slow", 26)
    macd_signal = macd_cfg.get("signal", 9)

    # Basic returns
    df["return"] = close.pct_change()
    df["log_return"] = np.log(
        close.replace(0, np.nan) / close.shift(1).replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)


    # SMAs / EMAs / WMAs
    for w in sma_windows:
        df[f"sma_{w}"] = close.rolling(w).mean()

    for w in ema_windows:
        df[f"ema_{w}"] = ema(close, w)

    for w in [10, 20]:
        df[f"wma_{w}"] = wma(close, w)

    # RSI
    for p in rsi_periods:
        df[f"rsi_{p}"] = rsi(close, p)

    # MACD
    macd_line, macd_sig, macd_hist = macd(close, macd_fast, macd_slow, macd_signal)
    df["macd"] = macd_line
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist

    # Bollinger
    (
        df["bb_upper"],
        df["bb_middle"],
        df["bb_lower"],
        df["bb_width"],
        df["bb_percent_b"],
    ) = bollinger_features(close, n=20, num_std=2)

    # High/low/open dependent indicators
    if high is not None and low is not None and open_ is not None:
        df["atr_14"] = atr(high, low, close, 14)
        adx_val, plus_di, minus_di = adx(high, low, close, 14)
        df["adx_14"] = adx_val
        df["plus_di_14"] = plus_di
        df["minus_di_14"] = minus_di

        k, d = stochastic_oscillator(high, low, close)
        df["stoch_k"] = k
        df["stoch_d"] = d

        df["williams_r_14"] = williams_r(high, low, close, 14)
        df["cci_20"] = cci(high, low, close, 20)
        df["parkinson_20"] = parkinson_vol(high, low, 20)
        df["garman_klass_20"] = garman_klass_vol(open_, high, low, close, 20)
        df = candle_features(df)
    else:
        print("  [INFO] Missing open/high/low: skipping ATR/ADX/Stoch/CCI/candles.")

    # Volume-based
    if volume is not None:
        df["obv"] = obv(close, volume)
        df["vpt"] = vpt(close, volume)
        df["volume_mean_20"] = volume.rolling(20).mean()
        df["volume_std_20"] = volume.rolling(20).std()
        df["volume_zscore_20"] = (volume - df["volume_mean_20"]) / df["volume_std_20"]
    else:
        print("  [INFO] No volume column: skipping OBV/VPT/volume features.")

    # Realized vol & multi-horizon returns
    for name, series in realized_vol(close).items():
        df[name] = series
    for name, series in multi_returns(close).items():
        df[name] = series

    # Legacy simple vol windows
    df["vol_7"] = df["return"].rolling(7).std()
    df["vol_30"] = df["return"].rolling(30).std()
    df["vol_90"] = df["return"].rolling(90).std()

    return df


def process_directory(raw_dir: str, processed_dir: str, ind_cfg: dict):
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"[WARN] Raw directory not found: {raw_dir}")
        return

    files = list(raw_path.glob("*.parquet"))
    print(f"[INFO] Found {len(files)} files in {raw_dir}")

    for f in files:
        print(f"[INFO] Processing: {f.name}")
        try:
            df = pd.read_parquet(f)

            # Normalize column names to lowercase
            df.columns = [str(c).lower() for c in df.columns]

            if "date" not in df.columns:
                raise RuntimeError("Missing 'date' column in raw file.")

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

            df_ind = compute_indicators_for_frame(df, ind_cfg)

            # Standardize date column name to 'Date' for downstream steps
            df_ind = df_ind.rename(columns={"date": "Date"})

            out_path = proc_path / f.name
            df_ind.to_parquet(out_path, index=False)
            print(f"[OK] Saved → {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed on {f.name}: {e}")


def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    sources = data_cfg["sources"]
    processed = data_cfg["processed"]
    ind_cfg = cfg.get("indicators", {})

    groups = [
        ("Companies", sources["companies_folder"], processed["companies_folder"]),
        ("Macro",     sources["macro_folder"],     processed["macro_folder"]),
        ("Market",    sources["market_folder"],    processed["market_folder"]),
    ]

    for name, raw_dir, proc_dir in groups:
        print(f"\n========== {name} ==========")
        process_directory(raw_dir, proc_dir, ind_cfg)

    print("\n[COMPLETE] All indicators computed.\n")


if __name__ == "__main__":
    main()
