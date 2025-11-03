#!/usr/bin/env python3
"""
Compute SMA, EMA, RSI, MACD, Bollinger Band Width (and ATR) for cached OHLCV parquet(s).
- Input: one file via --infile OR a directory via --indir (defaults to data/cache)
- Output: enriched parquet(s) with indicators in data/cache (versioned) + optional alias.
"""

from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd

# local import
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.append(str(ROOT_DIR))
from features.indicators import add_technical_indicators, IndicatorConfig  # noqa: E402

DEFAULT_CACHE = ROOT_DIR / "data" / "cache"

def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    df.columns = df.columns.str.lower()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _versioned_out_name(in_path: Path, suffix="_ind") -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d")
    base = in_path.stem
    return in_path.with_name(f"{base}{suffix}_{stamp}.parquet")

def compute_for_file(in_path: Path, cfg: IndicatorConfig, alias_out: Path | None) -> Path:
    df = _read_file(in_path)
    # if there is no ticker column, add one from filename (nice to have)
    if "ticker" not in df.columns:
        df["ticker"] = in_path.stem.split("_")[0]  # crude but useful (e.g., DSEX_20250101.parquet -> DSEX)
    enriched = add_technical_indicators(df, cfg)
    out_path = _versioned_out_name(in_path)
    enriched.to_parquet(out_path, index=True)  # keep index to preserve (ticker, date) if set
    print(f"[OK] Indicators saved → {out_path} ({len(enriched)} rows)")
    if alias_out:
        alias_out.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(alias_out, index=True)
        print(f"[OK] Alias parquet updated → {alias_out}")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, help="Single parquet file to process")
    ap.add_argument("--indir", type=str, default=str(DEFAULT_CACHE), help="Directory containing OHLCV parquets")
    ap.add_argument("--pattern", type=str, default="*.parquet", help="Glob to match input parquets")
    ap.add_argument("--alias-out", type=str, default=None, help="Optional alias parquet path to overwrite")
    ap.add_argument("--drop-warmup", action="store_true", default=True, help="Drop initial warmup rows")
    ap.add_argument("--keep-warmup", action="store_true", help="Keep NaN warmup rows instead of dropping")
    args = ap.parse_args()

    if args.keep_warmup:
        drop_warmup = False
    else:
        drop_warmup = True

    cfg = IndicatorConfig(drop_warmup=drop_warmup)

    if args.infile:
        in_path = Path(args.infile).resolve()
        if not in_path.exists():
            sys.exit(f"❌ infile not found: {in_path}")
        alias = Path(args.alias_out).resolve() if args.alias_out else None
        compute_for_file(in_path, cfg, alias)
        return

    # directory mode
    indir = Path(args.indir).resolve()
    if not indir.exists() or not indir.is_dir():
        sys.exit(f"❌ indir not found or not a directory: {indir}")

    paths = sorted(indir.glob(args.pattern))
    if not paths:
        sys.exit(f"❌ no parquet files found in: {indir} (pattern={args.pattern})")

    for p in paths:
        try:
            compute_for_file(p, cfg, alias_out=None)
        except Exception as e:
            print(f"[WARN] Failed {p.name}: {e}")

if __name__ == "__main__":
    main()


# python compute_indicators.py --infile ../data/cache/1JANATAMF_hist.csv --alias-out ../data/cache/1JANATAMF_with_indicators.parquet
