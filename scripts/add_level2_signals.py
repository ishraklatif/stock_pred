#!/usr/bin/env python3
"""
add_level2_signals.py

Adds Level-2 macro signals:
- VIX (^VIX)
- DXY (DX-Y.NYB or DX=F)
- NASDAQ 100 (^NDX)
- US 10Y yield (^TNX)
- US 2Y yield (^IRX)
- Copper (HG=F)
- Commodity index (DBC)

Downloads → Computes indicators → Merges with macro dataset.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import os

MACRO_PATH = "data/level1/macro_enriched.parquet"
OUTPUT_PATH = "data/level2/macro_level2.parquet"

LEVEL2_SYMBOLS = {
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",    # If fails, fallback to DX=F
    "NDX": "^NDX",
    "US10Y": "^TNX",
    "US2Y": "^IRX",
    "COPPER": "HG=F",
    "COMMODITY_IDX": "DBC"
}


def safe_download(symbol, start="2000-01-01"):
    """Safe download wrapper with fallback for DXY."""
    try:
        df = yf.download(symbol, start=start)
        if df.empty and symbol == "DX-Y.NYB":
            print("[WARN] Falling back to DX=F for DXY...")
            return yf.download("DX=F", start=start)
        return df
    except:
        print(f"[ERROR] Could not download: {symbol}")
        return None

def flatten_columns(df):
    """
    Flatten MultiIndex or tuple columns into simple string columns.
    Example: ('Close','HG=F') -> 'Close_HG=F'
    """
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            col = "_".join([str(c) for c in col if c not in (None, "")])
        new_cols.append(col)
    df.columns = new_cols
    return df

def normalize_price_columns(df):
    """
    Ensure df has a standard 'Close' column.
    Handles cases where Yahoo returns only 'Adj Close', 'Last Price', or other variations.
    """
    cols = df.columns

    # If Close exists, we are good
    if "Close" in cols:
        return df

    # If Adj Close exists, rename to Close
    if "Adj Close" in cols:
        df["Close"] = df["Adj Close"]
        return df
    
    # Futures & yields sometimes use this
    if "Last Price" in cols:
        df["Close"] = df["Last Price"]
        return df

    # If only one price-like column exists (rare)
    price_candidates = [c for c in cols if any(k in c.lower() for k in ["close", "price", "last", "value"])]
    if price_candidates:
        df["Close"] = df[price_candidates[0]]
        return df

    # As absolute fallback: pick ANY numeric column as Close
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    if len(numeric_cols) > 0:
        df["Close"] = df[numeric_cols[0]]
        print("[WARN] Fallback: using first numeric column as Close")
        return df

    raise RuntimeError("No usable price column found in dataframe.")


def add_indicators(df, prefix):
    # Simple returns
    df[f"{prefix}_return"] = df["Close"].pct_change()

    # Safe log returns
    ratio = df["Close"] / df["Close"].shift(1)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    df[f"{prefix}_log_return"] = np.log(ratio)

    # Moving averages
    df[f"{prefix}_SMA_10"] = df["Close"].rolling(10).mean()
    df[f"{prefix}_SMA_30"] = df["Close"].rolling(30).mean()

    df[f"{prefix}_EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df[f"{prefix}_EMA_30"] = df["Close"].ewm(span=30, adjust=False).mean()

    return df



def load_macro():
    if not os.path.exists(MACRO_PATH):
        raise RuntimeError("Macro enriched file not found. Run add_calendar_features.py first.")
    df = pd.read_parquet(MACRO_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def main():
    print("[INFO] Loading macro dataset...")
    macro = load_macro()

    print("[INFO] Downloading Level-2 signals...")
    level2_frames = []

    for name, symbol in LEVEL2_SYMBOLS.items():
        print(f"[INFO] Fetching {name}: {symbol}")
        df = safe_download(symbol)
        if df is None or df.empty:
            print(f"[WARN] Skipping {name}")
            continue

        df = df.reset_index()

        # --- NEW: flatten MultiIndex columns ---
        df = flatten_columns(df)

        df = normalize_price_columns(df) 


        # Ensure proper Date column
        df["Date"] = pd.to_datetime(df["Date"])

        # Add indicators
        df = add_indicators(df, prefix=name)

        # Prefix all columns except Date
        rename_map = {}
        for col in df.columns:
            if col != "Date":
                rename_map[col] = f"{name}_{col}"
        df = df.rename(columns=rename_map)


        level2_frames.append(df)

    # Merge Level-2 signals into macro
    merged = macro.copy()

    for df in level2_frames:
        merged = pd.merge(merged, df, how="outer", on="Date")

    merged = merged.sort_values("Date").ffill()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    merged.to_parquet(OUTPUT_PATH)
    print(f"[OK] Saved Level-2 enriched macro dataset → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

# python3 scripts/add_level2_signals.py
