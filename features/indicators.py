import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class IndicatorConfig:
    drop_warmup: bool = True
    sma_windows: tuple = (14, 21, 50)
    ema_windows: tuple = (14, 21, 50)
    rsi_window: int = 14
    bb_window: int = 20
    bb_num_std: int = 2
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_window: int = 14


def sma(series, window):
    return series.rolling(window).mean()


def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()


def rsi(series, window):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series, fast, slow, signal):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df, window):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def add_technical_indicators(df: pd.DataFrame, cfg: IndicatorConfig):
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # --- SMA ---
    for w in cfg.sma_windows:
        df[f"sma_{w}"] = sma(df["close"], w)

    # --- EMA ---
    for w in cfg.ema_windows:
        df[f"ema_{w}"] = ema(df["close"], w)

    # --- RSI ---
    df[f"rsi_{cfg.rsi_window}"] = rsi(df["close"], cfg.rsi_window)

    # --- MACD ---
    macd_line, signal_line, histogram = macd(
        df["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
    )
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_diff"] = histogram

    # --- Bollinger Bandwidth ---
    mid = sma(df["close"], cfg.bb_window)
    std = df["close"].rolling(cfg.bb_window).std()
    upper = mid + cfg.bb_num_std * std
    lower = mid - cfg.bb_num_std * std
    df[f"bb_bw_{cfg.bb_window}_{cfg.bb_num_std}"] = (upper - lower) / mid

    # --- ATR ---
    df[f"atr_{cfg.atr_window}"] = atr(df, cfg.atr_window)

    # --- Drop warmup rows (first 50 rows may contain NaN) ---
    if cfg.drop_warmup:
        warmup = max(cfg.sma_windows + cfg.ema_windows + (cfg.rsi_window, cfg.bb_window, cfg.atr_window))
        df = df.dropna().reset_index(drop=True)

    return df
