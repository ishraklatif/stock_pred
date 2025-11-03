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
    parser.add_argument("--excel", "-e", required=True, help="Path to Excel file with symbols (e.g., data/shares.xlsx).")
    parser.add_argument("--sheet", "-s", default=None, help="Excel sheet name (optional).")
    parser.add_argument("--column", "-c", default=None, help="Column name containing symbols (auto-detect if omitted).")
    parser.add_argument("--outdir", "-o", default=None, help="Output directory. Default is project ROOT/data.")
    parser.add_argument("--dryrun", "-n", action="store_true", help="Only list candidate functions; do not execute.")
    args = parser.parse_args()

    # Load symbols (limit = 5 by requirement)
    symbols = load_symbols_from_excel(args.excel, args.column, args.sheet, limit=5)
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


if __name__ == "__main__":
    main()
