# #!/usr/bin/env python3
# """
# Compute SMA, EMA, RSI, MACD, Bollinger Band Width, ATR, and more
# for cached OHLCV files (CSV or Parquet).

# Works with:
# - Index parquet files (AXJO, SPY, FTSE, N225)
# - Company-level CSV/Parquet

# Output:
#   <basename>_ind_<YYYYMMDD>.parquet
# """

# from __future__ import annotations
# import argparse
# from datetime import datetime
# from pathlib import Path
# import sys
# import pandas as pd

# SCRIPT_DIR = Path(__file__).resolve().parent
# ROOT_DIR = SCRIPT_DIR.parent
# sys.path.append(str(ROOT_DIR))

# from features.indicators import add_technical_indicators, IndicatorConfig  # noqa: E402

# DEFAULT_CACHE = ROOT_DIR / "data" / "cache"


# # ------------------------------
# # Helpers
# # ------------------------------
# def _read_file(path: Path) -> pd.DataFrame:
#     """Load CSV/Parquet and return a df with a proper datetime 'date' column."""
#     if path.suffix.lower() == ".csv":
#         df = pd.read_csv(path)
#     else:
#         df = pd.read_parquet(path)

#     # normalize column names
#     df.columns = df.columns.str.lower()

#     # CASE 1: date already a column
#     if "date" in df.columns:
#         df["date"] = pd.to_datetime(df["date"], errors="coerce")
#         df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
#         return df

#     # CASE 2: date in datetime index
#     if isinstance(df.index, pd.DatetimeIndex):
#         df = df.reset_index().rename(columns={"index": "date"})
#         df["date"] = pd.to_datetime(df["date"], errors="coerce")
#         df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
#         return df

#     # CASE 3: index name looks like date
#     if df.index.name and "date" in df.index.name.lower():
#         df = df.reset_index().rename(columns={df.index.name: "date"})
#         df["date"] = pd.to_datetime(df["date"], errors="coerce")
#         df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
#         return df

#     raise ValueError(f"❌ No date found in {path}. Columns={df.columns.tolist()}")


# def _extract_ticker(path: Path) -> str:
#     """Get ticker name from filename, e.g. CBA.AX_hist.csv → CBA.AX"""
#     base = path.stem
#     return base.split("_")[0]


# def _versioned_out_name(path: Path, suffix="_ind") -> Path:
#     stamp = datetime.utcnow().strftime("%Y%m%d")
#     base = path.stem
#     return path.with_name(f"{base}{suffix}_{stamp}.parquet")


# # ------------------------------
# # Core: compute indicators
# # ------------------------------
# def compute_for_file(path: Path, cfg: IndicatorConfig, alias_out: Path | None) -> Path:
#     df = _read_file(path)

#     # basic sanity
#     required = ["date", "open", "high", "low", "close", "volume"]
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise ValueError(f"❌ Missing required columns {missing} in {path}")

#     # add ticker
#     df["ticker"] = _extract_ticker(path)

#     # compute indicators
#     enriched = add_technical_indicators(df, cfg)

#     # ---- ensure 'date' is a proper column ----
#     if "date" in enriched.columns:
#         enriched["date"] = pd.to_datetime(enriched["date"], errors="coerce")
#     elif isinstance(enriched.index, pd.DatetimeIndex):
#         # bring index back as 'date'
#         enriched = enriched.reset_index().rename(columns={"index": "date"})
#         enriched["date"] = pd.to_datetime(enriched["date"], errors="coerce")
#     else:
#         # last resort: copy by position (avoid index alignment issues)
#         enriched["date"] = df["date"].values

#     enriched = enriched.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

#     # save
#     out_path = _versioned_out_name(path)
#     enriched.to_parquet(out_path, index=False)
#     print(f"[OK] Indicators saved → {out_path} ({len(enriched)} rows)")

#     if alias_out:
#         alias_out.parent.mkdir(parents=True, exist_ok=True)
#         enriched.to_parquet(alias_out, index=False)
#         print(f"[OK] Alias updated → {alias_out}")

#     return out_path


# # ------------------------------
# # Main
# # ------------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--infile", type=str, help="Single CSV/parquet to process")
#     ap.add_argument("--indir", type=str, default=str(DEFAULT_CACHE), help="Directory with OHLCV files")
#     ap.add_argument("--pattern", type=str, default="*.parquet", help="Glob pattern (e.g., *.parquet)")
#     ap.add_argument("--alias-out", type=str, default=None, help="Optional alias parquet path")
#     ap.add_argument("--keep-warmup", action="store_true", help="Keep NaN warmup rows instead of dropping")
#     args = ap.parse_args()

#     cfg = IndicatorConfig(drop_warmup=not args.keep_warmup)

#     # Single file mode
#     if args.infile:
#         in_path = Path(args.infile).resolve()
#         if not in_path.exists():
#             sys.exit(f"❌ infile not found: {in_path}")
#         alias = Path(args.alias_out).resolve() if args.alias_out else None
#         compute_for_file(in_path, cfg, alias)
#         return

#     # Directory mode
#     indir = Path(args.indir).resolve()
#     if not indir.exists() or not indir.is_dir():
#         sys.exit(f"❌ indir not found or not a directory: {indir}")

#     paths = sorted(indir.glob(args.pattern))
#     if not paths:
#         sys.exit(f"❌ no files found in: {indir} (pattern={args.pattern})")

#     print(f"[INFO] Found {len(paths)} files to process in {indir}")
#     for p in paths:
#         try:
#             compute_for_file(p, cfg, alias_out=None)
#         except Exception as e:
#             print(f"[WARN] Failed {p.name}: {e}")

#     print("[DONE]")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
compute_indicators.py

A clean, dependency-free indicator calculator.
No pandas_ta required — only pandas & numpy.

Reads raw files from:
    data/raw/<symbol>.parquet

Writes processed files to:
    data/processed/<symbol>.parquet
"""

import os
import yaml
import numpy as np
import pandas as pd

CONFIG_PATH = "config/config.yaml"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def flatten_dict_lists(d):
    combined = []
    for v in d.values():
        if isinstance(v, list):
            combined.extend(v)
    return combined


def gather_all_symbols(cfg):
    symbols = []

    if cfg["include"]["markets"]:
        symbols.extend(flatten_dict_lists(cfg["markets"]))

    if cfg["include"]["commodities"]:
        symbols.extend(cfg["commodities"])

    if cfg["include"]["fx"]:
        symbols.extend(cfg["fx"])

    return list(dict.fromkeys(symbols))


# ------------------------------
#  INDICATOR FUNCTIONS
# ------------------------------

def rsi(close, period=14):
    delta = close.diff()

    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))


def ema(close, span):
    return close.ewm(span=span, adjust=False).mean()


def macd(close):
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist


def bollinger(close, n=20, num_std=2):
    ma = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = ma + std * num_std
    lower = ma - std * num_std
    return upper, ma, lower


def atr(high, low, close, period=14):
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ------------------------------
#  COMPUTE INDICATORS
# ------------------------------

def compute_indicators(df):
    df = df.copy()

    df["return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # MAs
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()
    df["EMA_10"] = ema(df["Close"], 10)
    df["EMA_30"] = ema(df["Close"], 30)

    # RSI
    df["RSI_14"] = rsi(df["Close"], 14)

    # MACD
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"])

    # Bollinger Bands
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = bollinger(df["Close"])

    # ATR
    df["ATR_14"] = atr(df["High"], df["Low"], df["Close"])

    # Volatility windows
    df["vol_7"] = df["return"].rolling(7).std()
    df["vol_30"] = df["return"].rolling(30).std()
    df["vol_90"] = df["return"].rolling(90).std()

    return df


# ------------------------------
#  MAIN
# ------------------------------

def main():
    cfg = load_config()
    symbols = gather_all_symbols(cfg)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"[INFO] Computing indicators for {len(symbols)} symbols...")

    for symbol in symbols:
        input_path = f"{RAW_DIR}/{symbol}.parquet"
        if not os.path.exists(input_path):
            print(f"[WARN] Missing raw file: {symbol}")
            continue

        df = pd.read_parquet(input_path)
        if df.empty:
            print(f"[WARN] Empty file: {symbol}")
            continue

        print(f"[INFO] Processing {symbol}...")
        df = compute_indicators(df)
        df.to_parquet(f"{PROCESSED_DIR}/{symbol}.parquet")

        print(f"[OK] Saved indicators for {symbol}")

    print("[COMPLETE] All indicators computed.")


if __name__ == "__main__":
    main()
# python3 scripts/compute_indicators.py
