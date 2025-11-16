#!/usr/bin/env python3
"""
Clean Minimal Fetching Script for StockPred

✔ fetch_local_index() → DSEX using bdshare only
✔ fetch_company_history() → OHLCV for each company using get_basic_hist_data()
✔ fetch_foreign_indices() → SPY, FTSE, N225 using yfinance ONLY (NOT for DSEX)
✔ versioned caching
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (ROOT_DIR / "data").resolve()
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------

def save_versioned(df: pd.DataFrame, name: str):
    """Save df to data/cache/name_YYYYMMDD.parquet"""
    stamp = datetime.utcnow().strftime("%Y%m%d")
    fname = f"{name.replace('^','')}_{stamp}.parquet"
    out = CACHE_DIR / fname
    df.to_parquet(out, index=False)
    print(f"[INFO] Cached {name} → {out}")
    return out


def load_csv_fallback(csv_path: str):
    p = Path(csv_path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["date"])
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"[WARN] CSV fallback failed: {e}")
        return None

# --------------------------------------------------------
# Fetch DSEX (NO yfinance)
# --------------------------------------------------------

def fetch_local_index(
    ticker="^DSEX",
    csv_path=None,
    index_source="auto"
):
    """
    Fetch DSEX using ONLY:
      1. bdshare.get_dsex_data()
      2. bdshare.get_hist_data("DSEX")
      3. CSV fallback

    No yfinance for local index.
    """

    import bdshare

    print(f"[INFO] Fetching DSEX (no yfinance) using source='{index_source}'")
    df = None

    # 1. get_dsex_data()
    if index_source in ("auto", "dsex"):
        try:
            print("[INFO] Trying bdshare.get_dsex_data()...")
            df = bdshare.get_dsex_data()
            if df is not None and not df.empty:
                print("[SUCCESS] Loaded via get_dsex_data()")
        except Exception as e:
            print(f"[WARN] get_dsex_data() failed: {e}")

    # 2. get_hist_data("DSEX")
    if (df is None or df.empty) and index_source in ("auto", "hist"):
        try:
            print("[INFO] Trying bdshare.get_hist_data('DSEX')...")
            df = bdshare.get_hist_data("DSEX")
            if df is not None and not df.empty:
                print("[SUCCESS] Loaded via get_hist_data('DSEX')")
        except Exception as e:
            print(f"[WARN] get_hist_data('DSEX') failed: {e}")

    # 3. CSV fallback
    if (df is None or df.empty) and csv_path:
        print(f"[INFO] Falling back to CSV → {csv_path}")
        df = load_csv_fallback(csv_path)

    # Final failure
    if df is None or df.empty:
        raise RuntimeError("❌ Could not fetch DSEX from bdshare or CSV.")

    # Clean
    df.columns = [c.lower().strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    save_versioned(df, ticker)
    return df

# --------------------------------------------------------
# Foreign indices (yfinance ONLY here)
# --------------------------------------------------------

def fetch_foreign_indices(tickers, months=6):
    """
    Fetch global indices via yfinance ONLY
    Example: ["SPY", "^FTSE", "^N225"]
    """
    import yfinance as yf

    for t in tickers:
        print(f"[INFO] Fetching {t} from yfinance...")
        df = yf.download(t, period=f"{months}mo", progress=False)

        if df.empty:
            print(f"[WARN] {t} returned empty")
            continue

        df = df.reset_index().rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        save_versioned(df, t)

# --------------------------------------------------------
# Company historical (BDShare)
# --------------------------------------------------------

def fetch_company_history(excel, column, sheet, months, outdir, limit=None):
    import datetime as dt
    from bdshare import get_basic_hist_data

    df_excel = pd.read_excel(excel, sheet_name=sheet or 0)
    symbols = (
        df_excel[column]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if limit:
        symbols = symbols[:limit]

    print(f"[INFO] Loaded {len(symbols)} symbols from Excel")

    start = dt.datetime.now().date() - dt.timedelta(days=months * 30)
    end = dt.datetime.now().date()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {"run_at": datetime.now().strftime("%Y%m%d_%H%M%S"), "symbols": [], "errors": []}

    for sym in symbols:
        try:
            print(f"[INFO] Fetching {sym} ({start} → {end})...")
            df = get_basic_hist_data(start, end, sym)
            if df is None or df.empty:
                raise ValueError("Empty dataframe")

            df.to_csv(outdir / f"{sym}_hist.csv", index=False)
            manifest["symbols"].append(sym)
        except Exception as e:
            print(f"[WARN] {sym}: {e}")
            manifest["errors"].append({"symbol": sym, "error": str(e)})

    json.dump(manifest, open(outdir / "manifest.json", "w"), indent=2)
    print("[DONE] Company historical fetch complete")

# --------------------------------------------------------
# CLI
# --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean StockPred Fetcher")

    parser.add_argument("--mode", choices=["local-index", "historical", "foreign"], default="local-index")

    # DSEX
    parser.add_argument("--ticker", default="^DSEX")
    parser.add_argument("--csv-fallback", default=str(DATA_DIR / "dsex_fallback.csv"))
    parser.add_argument("--index-source", choices=["auto","dsex","hist","csv"], default="auto")

    # Company OHLCV
    parser.add_argument("--excel")
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--column", default="TRADING CODE")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--outdir", default=str(CACHE_DIR))

    # Foreign indices
    parser.add_argument("--foreign", nargs="*", default=["SPY","^FTSE","^N225"])

    args = parser.parse_args()
    print(f"[MODE] {args.mode}")

    if args.mode == "local-index":
        fetch_local_index(
            ticker=args.ticker,
            csv_path=args.csv_fallback,
            index_source=args.index_source
        )

    elif args.mode == "foreign":
        fetch_foreign_indices(args.foreign, months=args.months)

    elif args.mode == "historical":
        if not args.excel:
            parser.error("--excel required for historical mode")
        fetch_company_history(
            excel=args.excel,
            column=args.column,
            sheet=args.sheet,
            months=args.months,
            outdir=args.outdir,
            limit=args.limit
        )

    else:
        parser.error(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()



