#!/usr/bin/env python3
"""
fetch_all_bdshare_funcs.py

Read share symbols from an Excel file, introspect `bdshare` to discover functions that
accept a single 'symbol-like' argument, and run those functions for each symbol.
Results are saved as CSVs (one file per function) in the project root `data/` directory
(../data relative to this script), unless an absolute --outdir is provided.

Usage examples:
  python fetch_all_bdshare_funcs.py \
    --excel ../data/shares.xlsx \
    --sheet Stocks \
    --column Symbol

  # From project root if your Excel is data/shares.xlsx:
  python scripts/fetch_all_bdshare_funcs.py --excel data/shares.xlsx --column Symbol

Notes:
- The script is conservative: it only auto-calls functions that appear to accept a
  single required 'symbol-like' parameter (e.g., symbol, ticker, code, instrument, name).
- It skips private functions and those with extra required parameters.
- All exceptions are captured and summarized in a manifest JSON.
"""

import argparse
import json
import sys
import inspect
import importlib
import pkgutil
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Path setup (bullet-proof; always saves to project root /data by default)
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent          # .../stock_pred/scripts
ROOT_DIR   = SCRIPT_DIR.parent                        # .../stock_pred
DATA_DIR   = (ROOT_DIR / "data").resolve()            # .../stock_pred/data
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Uncomment for quick sanity debug:
# print(f"[DEBUG] SCRIPT_DIR = {SCRIPT_DIR}")
# print(f"[DEBUG] ROOT_DIR   = {ROOT_DIR}")
# print(f"[DEBUG] DATA_DIR   = {DATA_DIR}")


# ------------------------------
# Config
# ------------------------------
SYMBOL_PARAM_CANDIDATES = {
    "symbol", "symbols", "scrip", "scrips", "code", "codes",
    "ticker", "tickers", "instrument", "instruments",
    "name", "names", "share", "shares"
}
# If a function has these required params in addition to a "symbol-like" parameter, we skip it.
DISALLOWED_REQUIRED_PARAMS = {"provider", "session", "client", "api_key"}

# Some functions may return very wide/long dataframes; limit printing/log verbosity
MAX_SHOW_ROWS = 5

# ─────────────────────────────────────────────────────────
# Local index fetchers (OpenBB → yfinance → CSV)
# ─────────────────────────────────────────────────────────
from datetime import timedelta

try:
    from openbb import obb
    HAS_OPENBB = True
except Exception:
    HAS_OPENBB = False

try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    HAS_YF = False


def _save_versioned_parquet(df: pd.DataFrame, ticker: str) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d")
    fname = f"{ticker.replace('^','')}_{stamp}.parquet"
    out = DATA_DIR / "cache" / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[INFO] Cached {ticker} → {out}")
    return str(out)


def fetch_foreign_indices(tickers=["SPY", "^FTSE", "^N225"], months=6):
    from datetime import datetime, timedelta
    start = datetime.utcnow() - timedelta(days=months * 30)
    end = datetime.utcnow()

    data_map = {}

    for t in tickers:
        print(f"[INFO] Fetching {t} ...")
        df = None

        # Try OpenBB first
        try:
            from openbb import obb
            df = obb.equity.price.historical(
                ticker=t, start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d")
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.reset_index(inplace=True)
                df.rename(columns={
                    "Date": "date", "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume"
                }, inplace=True, errors="ignore")
                df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            print(f"[WARN] OpenBB fetch failed for {t}: {e}")

        # Fallback to yfinance
        if df is None or df.empty:
            try:
                import yfinance as yf
                yf_df = yf.download(t, start=start, end=end, progress=False)
                if not yf_df.empty:
                    df = yf_df.reset_index().rename(columns={
                        "Date": "date", "Open": "open", "High": "high",
                        "Low": "low", "Close": "close", "Volume": "volume"
                    })
                    df["date"] = pd.to_datetime(df["date"])
            except Exception as e:
                print(f"[WARN] yfinance fetch failed for {t}: {e}")

        if df is None or df.empty:
            print(f"[ERROR] No data for {t}")
            continue

        df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
        _save_versioned_parquet(df, t)
        data_map[t] = df

    return data_map



def _load_csv_fallback(csv_path: str) -> Optional[pd.DataFrame]:
    p = Path(csv_path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["date"])
        cols = ["date", "open", "high", "low", "close", "volume"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"[WARN] CSV missing columns {missing}; attempting best-effort subset.")
        df = df[[c for c in cols if c in df.columns]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df
    except Exception as e:
        print(f"[WARN] Failed to read fallback CSV: {e}")
        return None


def fetch_local_index(ticker="^DSEX", months=3, csv_path=None):
    import bdshare, pandas as pd
    print(f"[INFO] Attempting to fetch DSEX index data from bdshare or fallback CSV")

    df = None
    try:
        # Try bdshare — but this only works for company symbols, not the DSEX index
        df = bdshare.get_hist_data("DSEX")
        if df is not None and not df.empty:
            print("[INFO] Loaded data via bdshare.get_hist_data('DSEX')")
    except Exception as e:
        print(f"[WARN] bdshare.get_hist_data('DSEX') failed: {e}")
        df = None

    # Fallback to CSV (most reliable for DSEX)
    if (df is None or df.empty) and csv_path:
        print(f"[INFO] Falling back to CSV → {csv_path}")
        df = _load_csv_fallback(csv_path)

    if df is None or df.empty:
        raise RuntimeError("❌ Failed to fetch DSEX time-series data: bdshare and CSV both unavailable.")

    df.columns = df.columns.str.lower().str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _save_versioned_parquet(df, ticker)
    return df








# ------------------------------
# Excel helpers
# ------------------------------
def resolve_excel_path(user_path: str) -> Path:
    """
    Resolve the Excel path robustly:
    - If absolute, use it as-is.
    - If relative, interpret it relative to the current working directory first.
      If not found, fall back to ROOT_DIR (project root).
    """
    p = Path(user_path)
    if p.is_absolute() and p.exists():
        return p
    # try relative to CWD
    if p.exists():
        return p.resolve()
    # fallback: relative to project root
    alt = (ROOT_DIR / p)
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(f"Excel file not found at '{user_path}' or '{alt}'")


def load_symbols_from_excel(path: str, column: Optional[str], sheet: Optional[str], limit: int = 5) -> List[str]:
    excel_path = resolve_excel_path(path)
    # Load Excel — default to the first sheet if not provided
    df_or_dict = pd.read_excel(excel_path, sheet_name=sheet or 0)

    # If multiple sheets were read (dict of DataFrames), take the first one
    if isinstance(df_or_dict, dict):
        first_sheet = list(df_or_dict.keys())[0]
        print(f"[INFO] Multiple sheets detected, using first sheet: {first_sheet}")
        df = df_or_dict[first_sheet]
    else:
        df = df_or_dict

    chosen_col = column
    if chosen_col is None:
        # Try common column names
        candidates = ["Symbol", "SYMBOL", "symbol", "Share", "ShareName", "Ticker", "Code", "Name", "Shares", "TRADING CODE"]
        for c in candidates:
            if c in df.columns:
                chosen_col = c
                break
        if chosen_col is None:
            # fallback: first object-dtype column
            for c in df.columns:
                if df[c].dtype == object:
                    chosen_col = c
                    break

    if chosen_col is None or chosen_col not in df.columns:
        raise ValueError(
            f"Could not detect a symbol column. Available columns: {list(df.columns)}. "
            f"Pass --column to choose one explicitly."
        )

    symbols = (
        df[chosen_col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    symbols = symbols[:limit]  # limit to first N (default 5)
    return symbols


# ------------------------------
# Introspection helpers
# ------------------------------
def iter_bdshare_modules(root_mod) -> Iterable[Any]:
    """Yield bdshare root module and any importable submodules."""
    yield root_mod
    if hasattr(root_mod, "__path__"):
        for m in pkgutil.walk_packages(root_mod.__path__, prefix=root_mod.__name__ + "."):
            try:
                sub = importlib.import_module(m.name)
                yield sub
            except Exception as e:
                sys.stderr.write(f"[WARN] Failed to import {m.name}: {e}\n")


def looks_symbol_function(func: Any) -> Optional[str]:
    """
    Return the *actual* parameter name to use (e.g., 'instrument', 'symbol', 'code', ...)
    if the function:
      - is public (no leading underscore)
      - is defined in a bdshare* module
      - has a parameter whose name is one of SYMBOL_PARAM_CANDIDATES
      - and has no other required parameters besides that symbol-like parameter.
    Otherwise return None.
    """
    if not inspect.isfunction(func):
        return None
    if func.__name__.startswith("_"):
        return None

    mod = getattr(func, "__module__", "") or ""
    if not mod.startswith("bdshare"):
        return None

    try:
        sig = inspect.signature(func)
    except Exception:
        return None

    params = list(sig.parameters.values())

    # Find any parameter whose name looks like a symbol param
    cand_param = None
    for p in params:
        pname = p.name.lower()
        if pname in SYMBOL_PARAM_CANDIDATES:
            cand_param = p
            break

    if cand_param is None:
        # No symbol-like parameter => not per-symbol
        return None

    # Collect required params
    required = [
        p for p in params
        if p.default is inspect._empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]

    # If there are required params other than the candidate one, skip
    other_required = [p for p in required if p is not cand_param]
    if other_required:
        return None

    # Guard against disallowed names being the *only* required parameter (rare)
    if cand_param.name in DISALLOWED_REQUIRED_PARAMS:
        return None

    # We can call this function per-symbol using cand_param.name
    return cand_param.name


def discover_symbol_functions() -> List[Tuple[str, Any, str]]:
    """
    Discover bdshare functions that *accept* a symbol-like parameter (required or optional),
    and have no other required parameters. Returns (qualified_name, function_obj, param_name).
    """
    try:
        bdshare = importlib.import_module("bdshare")
    except Exception as e:
        sys.stderr.write(f"[ERROR] Could not import bdshare: {e}\n")
        sys.exit(1)

    found: List[Tuple[str, Any, str]] = []
    seen = set()

    for mod in iter_bdshare_modules(bdshare):
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if id(obj) in seen:
                continue
            seen.add(id(obj))

            sym_param = looks_symbol_function(obj)
            if sym_param:
                qname = f"{mod.__name__}.{name}"
                found.append((qname, obj, sym_param))

    # IMPORTANT: do NOT add any fallback for generic get_*.
    return sorted(found, key=lambda x: x[0])


# ------------------------------
# Normalization helpers
# ------------------------------
def normalize_result(result: Any, symbol: str, func_name: str) -> pd.DataFrame:
    """
    Convert a variety of possible returns into a DataFrame with a Company column.
    - pandas DataFrame: add Company column
    - pandas Series: to_frame().T + Company
    - dict: DataFrame([{...}])
    - scalar/other: wrap into a DataFrame with columns ['value']
    """
    try:
        if hasattr(result, "to_frame") and not hasattr(result, "to_csv") and hasattr(result, "index"):
            # likely a Series
            df = result.to_frame().T
        elif hasattr(result, "to_csv"):
            # likely a DataFrame
            df = result.copy()
        elif isinstance(result, dict):
            df = pd.DataFrame([result])
        else:
            df = pd.DataFrame([{"value": result}])
        df["Company"] = symbol
        df["__source_function__"] = func_name
        return df
    except Exception as e:
        # Fallback: raw string
        return pd.DataFrame([{
            "Company": symbol,
            "__source_function__": func_name,
            "value": str(result),
            "__note__": f"Normalization fallback due to: {e}"
        }])


# ------------------------------
# Main run
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch all symbol-accepting bdshare functions for shares listed in Excel.")
    parser.add_argument("--excel", "-e", required=False, help="Path to Excel file with symbols (e.g., data/shares.xlsx).")
    parser.add_argument("--sheet", "-s", default=None, help="Excel sheet name (optional).")
    parser.add_argument("--column", "-c", default=None, help="Column name containing symbols (auto-detect if omitted).")
    parser.add_argument("--outdir", "-o", default=None, help="Output directory. Default is project ROOT/data.")
    parser.add_argument("--dryrun", "-n", action="store_true", help="Only list candidate functions; do not execute.")
    
    # CLI mode for US-1.3
    parser.add_argument("--mode", choices=["local-index", "bdshare-scan", "historical"], default="local-index",
                    help="local-index = fetch DSEX index, bdshare-scan = run all bdshare funcs, historical = fetch OHLCV per ticker")

    parser.add_argument("--ticker", default="^DSEX", help="Index ticker (default ^DSEX)")
    parser.add_argument("--months", type=int, default=3, help="How many months of history to fetch")
    parser.add_argument("--csv-fallback", default=str(ROOT_DIR / "data" / "dsex_fallback.csv"),
                        help="CSV fallback path for local index")
    parser.add_argument("--alias-out", default=str(DATA_DIR / "dsex.parquet"),
                        help="Optional stable alias parquet path (copied to) for easy downstream use")

    parser.add_argument("--limit", type=int, default=None,
    help="Limit number of symbols to fetch (default: all)")


    args = parser.parse_args()

    print(f"[MODE] Running in {args.mode.upper()} mode")

    if args.mode == "local-index":
        df = fetch_local_index(ticker=args.ticker, months=args.months, csv_path=args.csv_fallback)
        # Optional stable alias copy
        try:
            if args.alias_out:
                df.to_parquet(args.alias_out, index=False)
                print(f"[INFO] Wrote alias parquet → {args.alias_out}")
        except Exception as e:
            print(f"[WARN] Failed to write alias parquet: {e}")
        return

    elif args.mode == "bdshare-scan" and not args.excel:
        parser.error("--excel is required in bdshare-scan mode.")

    # Else keep your existing bdshare discovery flow:
    elif args.mode == "bdshare-scan":


        # Load symbols (limit = 5 by requirement)
        symbols = load_symbols_from_excel(args.excel, args.column, args.sheet, limit=args.limit)

        if not symbols:
            sys.stderr.write("[ERROR] No symbols loaded from Excel.\n")
            sys.exit(1)

        print(f"[INFO] Loaded {len(symbols)} symbols from {args.excel}: {symbols[:min(5, len(symbols))]}")

        # Discover bdshare functions
        funcs = discover_symbol_functions()
        if not funcs:
            sys.stderr.write("[ERROR] No bdshare functions found that accept a symbol-like parameter.\n")
            sys.exit(2)

        print(f"[INFO] Discovered {len(funcs)} candidate functions:")
        for qn, _, symparam in funcs:
            print(f"  - {qn}({symparam}=...)")

        if args.dryrun:
            print("[DRYRUN] Exiting without executing functions.")
            return

        # Determine the output directory (absolute)
        outdir_path = Path(args.outdir).resolve() if args.outdir else DATA_DIR
        outdir_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest: Dict[str, Any] = {
            "run_at": timestamp,
            "excel": str(resolve_excel_path(args.excel)),
            "sheet": args.sheet,
            "column": args.column,
            "outdir": str(outdir_path),
            "symbols_count": len(symbols),
            "functions_count": len(funcs),
            "functions": {},
        }

        # Execute each function for each symbol, write one CSV per function
        for qname, func, symbol_param in funcs:
            print(f"\n[RUN] {qname} for {len(symbols)} symbols...")

            per_func_frames = []
            per_func_status = {"success": [], "failure": []}

            for sym in symbols:
                try:
                    # Call function with keyword for the symbol param
                    result = func(**{symbol_param: sym})
                    df = normalize_result(result, sym, qname)
                    per_func_frames.append(df)
                    per_func_status["success"].append(sym)

                    # light preview
                    if hasattr(df, "head"):
                        _ = df.head(MAX_SHOW_ROWS)
                        print(f"  ✓ {sym}: {len(df)} rows (showing up to {MAX_SHOW_ROWS})")
                    else:
                        print(f"  ✓ {sym}: (non-DataFrame result)")

                except Exception as e:
                    per_func_status["failure"].append({"symbol": sym, "error": str(e)})
                    print(f"  ✗ {sym}: {e}")

            # Save a CSV if any success
            if per_func_frames:
                out_df = pd.concat(per_func_frames, ignore_index=True)
                # Sanitize filename
                safe_name = qname.replace(".", "_").replace("/", "_")
                csv_path = (outdir_path / f"{safe_name}__{timestamp}.csv")
                out_df.to_csv(csv_path, index=False)
                print(f"[OK] Wrote {len(out_df)} rows to {csv_path}")
                per_func_status["csv"] = str(csv_path)
            else:
                print(f"[WARN] No successful rows for {qname}")

            manifest["functions"][qname] = per_func_status

        # Write manifest (always alongside CSVs)
        manifest_path = (outdir_path / f"manifest_{timestamp}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Manifest written to {manifest_path}")
        print("[DONE]")
        return
    elif args.mode == "historical":
        import datetime as dt
        from bdshare import get_basic_hist_data

        if not args.excel:
            parser.error("--excel is required in historical mode (to read company tickers).")

        symbols = load_symbols_from_excel(args.excel, args.column, args.sheet, limit=args.limit)
        print(f"[INFO] Loaded {len(symbols)} symbols for historical fetch: {symbols}")

        start = dt.datetime.now().date() - dt.timedelta(days=args.months * 30)
        end = dt.datetime.now().date()

        outdir_path = Path(args.outdir or DATA_DIR / "cache").resolve()
        outdir_path.mkdir(parents=True, exist_ok=True)

        manifest = {"run_at": datetime.now().strftime("%Y%m%d_%H%M%S"), "symbols": [], "errors": []}

        for sym in symbols:
            try:
                print(f"[INFO] Fetching {sym} historical data ({start} → {end})...")
                df = get_basic_hist_data(start, end, sym)
                if df is None or df.empty:
                    raise ValueError("Empty dataframe returned")

                df.to_csv(outdir_path / f"{sym}_hist.csv", index=False)
                print(f"✅ Saved → {outdir_path / f'{sym}_hist.csv'} ({len(df)} rows)")
                manifest["symbols"].append(sym)

            except Exception as e:
                print(f"⚠️ {sym}: {e}")
                manifest["errors"].append({"symbol": sym, "error": str(e)})

        # Write manifest summary
        manifest_path = outdir_path / f"historical_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[DONE] Manifest saved → {manifest_path}")
        return
            

    else:
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()


#python fetch_data.py --mode bdshare-scan --excel data/shares.xlsx --column "TRADING CODE" --outdir ../data/cache --limit 1

#python fetch_data.py --mode historical --excel data/shares.xlsx --column "TRADING CODE" --outdir ../data/cache --months 1 --limit 1

