# features/indicators.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class IndicatorConfig:
    sma_windows: Tuple[int, ...] = (14, 21, 50)
    ema_windows: Tuple[int, ...] = (14, 21, 50)
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
    atr_window: int = 14
    drop_warmup: bool = True
    # column names
    col_open: str = "open"
    col_high: str = "high"
    col_low: str = "low"
    col_close: str = "close"
    col_volume: str = "volume"
    col_ticker: str = "ticker"
    col_date: str = "date"

def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False, min_periods=span).mean()

def _rsi_wilder(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> Dict[str, pd.Series]:
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_diff = macd_line - macd_signal
    return {"macd_line": macd_line, "macd_signal": macd_signal, "macd_diff": macd_diff}

def _bb_width(close: pd.Series, window: int, n_std: float) -> pd.Series:
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return (upper - lower) / mid  # normalized width

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

def _is_multi_index(df: pd.DataFrame, cfg: IndicatorConfig) -> bool:
    return isinstance(df.index, pd.MultiIndex) and cfg.col_ticker in df.index.names

def _ensure_index(df: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    df = df.copy()
    if _is_multi_index(df, cfg):
        return df.sort_index()
    if cfg.col_ticker in df.columns and cfg.col_date in df.columns:
        return df.set_index([cfg.col_ticker, cfg.col_date]).sort_index()
    if cfg.col_date in df.columns:
        return df.set_index(cfg.col_date).sort_index()
    return df.sort_index()

def _add_indicators_to_group(g: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    g = g.copy()

    # SMA
    for w in cfg.sma_windows:
        g[f"sma_{w}"] = g[cfg.col_close].rolling(w, min_periods=w).mean()
    # EMA
    for w in cfg.ema_windows:
        g[f"ema_{w}"] = _ema(g[cfg.col_close], w)
    # RSI
    g[f"rsi_{cfg.rsi_window}"] = _rsi_wilder(g[cfg.col_close], cfg.rsi_window)
    # MACD
    macd = _macd(g[cfg.col_close], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    for k, v in macd.items():
        g[k] = v
    # Bollinger Band Width
    g[f"bb_bw_{cfg.bb_window}_{int(cfg.bb_std)}"] = _bb_width(g[cfg.col_close], cfg.bb_window, cfg.bb_std)
    # ATR (bonus)
    g[f"atr_{cfg.atr_window}"] = _atr(g[cfg.col_high], g[cfg.col_low], g[cfg.col_close], cfg.atr_window)
    return g

def add_technical_indicators(df: pd.DataFrame, config: Optional[IndicatorConfig] = None) -> pd.DataFrame:
    cfg = config or IndicatorConfig()
    x = _ensure_index(df, cfg)
    if _is_multi_index(x, cfg):
        out = x.groupby(level=cfg.col_ticker, group_keys=False).apply(lambda g: _add_indicators_to_group(g, cfg))
    else:
        out = _add_indicators_to_group(x, cfg)

    if cfg.drop_warmup:
        warmup = max(
            max(cfg.sma_windows or (1,)),
            max(cfg.ema_windows or (1,)),
            cfg.rsi_window,
            cfg.macd_slow,
            cfg.bb_window,
            cfg.atr_window,
        )
        if _is_multi_index(out, cfg):
            out = out.groupby(level=cfg.col_ticker, group_keys=False).apply(lambda g: g.iloc[warmup-1:])
        else:
            out = out.iloc[warmup-1:]
    return out
