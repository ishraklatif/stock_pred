#!/usr/bin/env python3
"""
merge_all_features.py

Merges:

  - Company OHLC+indicators      (data/processed_companies)
  - Global macro indices         (data/processed_macro)
  - Macro-market instruments     (data/processed_macro_market)
  - Global calendar features     (data/processed_calendar/calendar_master.parquet)
  - News sentiment features      (data/news/sentiment)

into per-ticker TFT-ready parquet files:

    data/tft_ready/<SAFE_TICKER>.parquet

where SAFE_TICKER = ticker.replace(".", "_").replace("^", "").

Design goals:

  - No duplicate instruments (canonical mapping handled in fetch stage)
  - No double-prefixing (e.g. avoid "CSI300_CSI300_sent_mean")
  - Sentiment merged via safe_merge_sentiment:
        * only forward-fill *after* first valid date
        * no filling before first news
        * no filling after last news
  - Zero-variance features dropped per ticker,
    but 'Date', 'series', 'sector_id' are always kept.
"""

import os
from pathlib import Path

import yaml
import pandas as pd

CONFIG_PATH = "config/data.yaml"


# ======================================================================
# CONFIG
# ======================================================================

def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ======================================================================
# HELPERS
# ======================================================================

def safe_ticker_name(ticker: str) -> str:
    """Convert 'CBA.AX' → 'CBA_AX', '^GSPC' → 'GSPC', etc."""
    return ticker.replace(".", "_").replace("^", "")


def safe_merge_sentiment(df_price: pd.DataFrame, df_sent: pd.DataFrame) -> pd.DataFrame:
    """
    Correct sentiment merge:

      - Left join on Date
      - For each sentiment column:
            * forward-fill only AFTER first valid observation
            * do NOT fill before first_valid
            * do NOT fill beyond last_actual (natural consequence of ffill)
    """
    if df_sent is None or df_sent.empty:
        return df_price

    df = df_price.merge(df_sent, on="Date", how="left")

    # Identify sentiment columns
    sent_cols = [
        c for c in df.columns
        if "_sent_" in c
        or c.endswith("_pos")
        or c.endswith("_neu")
        or c.endswith("_neg")
    ]

    if not sent_cols:
        return df

    # Make sure df is sorted by Date so ffill respects time order
    df = df.sort_values("Date").reset_index(drop=True)

    for col in sent_cols:
        first_valid = df[col].first_valid_index()
        if first_valid is None:
            # Entire column is NaN (no news for this asset)
            continue

        # Forward-fill ONLY from first_valid onward
        df.loc[first_valid:, col] = df[col].loc[first_valid:].ffill()

    return df


def load_macro_block(processed_macro_dir: str) -> pd.DataFrame:
    """
    Load all processed macro frames and merge them on Date.

    Each file is expected to have columns:
        Date, open, high, low, close, volume, log_return,
        bb_percent_b, rsi_14, candle_body_ratio, obv

    We prefix non-Date columns with the canonical asset name.
    """
    macro_path = Path(processed_macro_dir)
    if not macro_path.exists():
        print(f"[WARN] Macro processed dir not found: {processed_macro_dir}")
        return None

    frames = []
    for f in sorted(macro_path.glob("*.parquet")):
        asset = f.stem  # e.g. "GSPC", "AXJO", "SSE", ...
        df = pd.read_parquet(f)
        if "Date" not in df.columns:
            # Old version: date lower-case
            if "date" in df.columns:
                df = df.rename(columns={"date": "Date"})
            else:
                print(f"[WARN] No Date column in macro file {f}, skipping.")
                continue

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        rename_map = {
            c: f"{asset}_{c}"
            for c in df.columns
            if c != "Date"
        }
        df = df.rename(columns=rename_map)
        frames.append(df)

    if not frames:
        return None

    macro_all = frames[0]
    for other in frames[1:]:
        macro_all = macro_all.merge(other, on="Date", how="outer")

    macro_all = macro_all.sort_values("Date").reset_index(drop=True)
    return macro_all


def load_market_block(processed_market_dir: str) -> pd.DataFrame:
    """
    Load all processed macro-market frames and merge them on Date,
    with asset-prefixed columns.
    """
    market_path = Path(processed_market_dir)
    if not market_path.exists():
        print(f"[WARN] Market processed dir not found: {processed_market_dir}")
        return None

    frames = []
    for f in sorted(market_path.glob("*.parquet")):
        asset = f.stem  # e.g. "DXY", "GOLD", "AUDUSD", ...
        df = pd.read_parquet(f)
        if "Date" not in df.columns:
            if "date" in df.columns:
                df = df.rename(columns={"date": "Date"})
            else:
                print(f"[WARN] No Date column in market file {f}, skipping.")
                continue

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        rename_map = {
            c: f"{asset}_{c}"
            for c in df.columns
            if c != "Date"
        }
        df = df.rename(columns=rename_map)
        frames.append(df)

    if not frames:
        return None

    market_all = frames[0]
    for other in frames[1:]:
        market_all = market_all.merge(other, on="Date", how="outer")

    market_all = market_all.sort_values("Date").reset_index(drop=True)
    return market_all


def load_calendar(calendar_dir: str) -> pd.DataFrame:
    """
    Load calendar_master.parquet from the processed_calendar folder.
    """
    cal_file = Path(calendar_dir) / "calendar_master.parquet"
    if not cal_file.exists():
        print(f"[WARN] Calendar file not found: {cal_file}")
        return None

    cal_df = pd.read_parquet(cal_file)
    if "Date" not in cal_df.columns:
        if "date" in cal_df.columns:
            cal_df = cal_df.rename(columns={"date": "Date"})
        else:
            raise RuntimeError("calendar_master has no 'Date' column.")

    cal_df["Date"] = pd.to_datetime(cal_df["Date"])
    cal_df = cal_df.sort_values("Date").reset_index(drop=True)
    return cal_df


def load_sentiment_block(sentiment_dir: str) -> pd.DataFrame:
    """
    Load all sentiment parquet files and merge them on Date.

    Each file is expected to have:
        Date, <ASSET>_sent_mean, <ASSET>_sent_max, ..., <ASSET>_pos, ...
    """
    sent_path = Path(sentiment_dir)
    if not sent_path.exists():
        print(f"[WARN] Sentiment dir not found: {sentiment_dir}")
        return None

    frames = []
    for f in sorted(sent_path.glob("*.parquet")):
        df = pd.read_parquet(f)
        if "Date" not in df.columns:
            # Some versions may have Date in the index
            if df.index.name == "Date":
                df = df.reset_index()
            elif "date" in df.columns:
                df = df.rename(columns={"date": "Date"})
            else:
                print(f"[WARN] No Date in sentiment file {f}, skipping.")
                continue

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        frames.append(df)

    if not frames:
        return None

    sent_all = frames[0]
    for other in frames[1:]:
        sent_all = sent_all.merge(other, on="Date", how="outer")

    sent_all = sent_all.sort_values("Date").reset_index(drop=True)
    return sent_all


def drop_zero_variance(df: pd.DataFrame):
    """
    Drop zero-variance columns, but always keep:
        - Date
        - series
        - sector_id
    """
    protected = {"Date", "series", "sector_id"}
    nunique = df.nunique()
    zero_var_cols = [
        c for c, n in nunique.items()
        if n <= 1 and c not in protected
    ]

    if zero_var_cols:
        df = df.drop(columns=zero_var_cols)
    return df, zero_var_cols


# ======================================================================
# MERGE PER TICKER
# ======================================================================

def merge_single_ticker(
    ticker: str,
    sector_name: str,
    companies_dir: str,
    macro_all: pd.DataFrame,
    market_all: pd.DataFrame,
    calendar_df: pd.DataFrame,
    sentiment_all: pd.DataFrame,
    out_dir: str,
):
    """
    Build a single merged DataFrame for one company ticker and save it.
    """
    safe_name = safe_ticker_name(ticker)
    in_path = Path(companies_dir) / f"{safe_name}.parquet"

    if not in_path.exists():
        print(f"[WARN] Company file missing for {ticker} at {in_path}, skipping.")
        return None

    print(f"\n[MERGE] {ticker}")

    df = pd.read_parquet(in_path)

    # Normalize Date
    if "Date" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "Date"})
        else:
            raise RuntimeError(f"{in_path} has no 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Add series & sector
    df["series"] = ticker          # group id (raw ticker string)
    df["sector_id"] = sector_name  # keep as sector name; TFT will handle as categorical

    # ------------------------------------------------------------------
    # Merge macro, market, calendar (simple left joins on Date)
    # ------------------------------------------------------------------
    if macro_all is not None:
        df = df.merge(macro_all, on="Date", how="left")

    if market_all is not None:
        df = df.merge(market_all, on="Date", how="left")

    if calendar_df is not None:
        df = df.merge(calendar_df, on="Date", how="left")

    # ------------------------------------------------------------------
    # Merge sentiment with safe forward-fill logic
    # ------------------------------------------------------------------
    df = safe_merge_sentiment(df, sentiment_all)

    # ------------------------------------------------------------------
    # Drop zero-variance columns (but keep Date/series/sector_id)
    # ------------------------------------------------------------------
    df, zero_var_cols = drop_zero_variance(df)
    if zero_var_cols:
        print(f"[INFO] Dropped zero-var cols for {ticker}: {len(zero_var_cols)}")

    # Final sort & save
    df = df.sort_values("Date").reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(out_dir) / f"{safe_name}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OK] Saved merged features → {out_path}")

    return df


# ======================================================================
# MAIN
# ======================================================================

def main():
    cfg = load_config()

    data_cfg = cfg["data"]
    companies_meta = cfg["companies"]["tickers_with_sectors"]

    processed_companies_dir = data_cfg["processed"]["companies_folder"]
    processed_macro_dir = data_cfg["processed"]["macro_folder"]
    processed_market_dir = data_cfg["processed"]["market_folder"]
    calendar_dir = data_cfg["processed"]["calendar_folder"]
    sentiment_dir = data_cfg["processed"]["news_sentiment_folder"]
    tft_ready_dir = data_cfg["processed"]["tft_ready_folder"] if "tft_ready_folder" in data_cfg["processed"] \
        else data_cfg["processed"].get("tft_ready_folder", "data/tft_ready")

    # --------------------------------------------------------------
    # Load global blocks once
    # --------------------------------------------------------------
    print("\n[LOAD] Macro block...")
    macro_all = load_macro_block(processed_macro_dir)

    print("\n[LOAD] Market block...")
    market_all = load_market_block(processed_market_dir)

    print("\n[LOAD] Calendar...")
    calendar_df = load_calendar(calendar_dir)

    print("\n[LOAD] Sentiment block...")
    sentiment_all = load_sentiment_block(sentiment_dir)

    # --------------------------------------------------------------
    # Merge per company
    # --------------------------------------------------------------
    companies_dir = processed_companies_dir
    out_dir = tft_ready_dir

    for ticker, sector_name in companies_meta.items():
        merge_single_ticker(
            ticker=ticker,
            sector_name=sector_name,
            companies_dir=companies_dir,
            macro_all=macro_all,
            market_all=market_all,
            calendar_df=calendar_df,
            sentiment_all=sentiment_all,
            out_dir=out_dir,
        )

    print("\n[MERGE COMPLETE] All company feature files generated.\n")


if __name__ == "__main__":
    main()
