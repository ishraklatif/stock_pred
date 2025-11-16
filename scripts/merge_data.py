# #!/usr/bin/env python3
# """
# Merge AXJO + SPY + FTSE + N225 indicator files into one aligned dataset.
# """

# import pandas as pd
# from pathlib import Path

# # Correct project root
# ROOT = Path(__file__).resolve().parent.parent
# CACHE = ROOT / "data" / "cache"
# OUTDIR = ROOT / "data" / "merged"
# OUTDIR.mkdir(parents=True, exist_ok=True)


# def load_latest(prefix: str) -> pd.DataFrame:
#     """Load the latest indicator parquet for a given prefix."""
#     files = sorted(CACHE.glob(f"{prefix}_*_ind_*.parquet"), reverse=True)
#     if not files:
#         raise FileNotFoundError(f"No indicator file found for {prefix}")

#     df = pd.read_parquet(files[0])

#     # Clean & normalize
#     df["date"] = pd.to_datetime(df["date"], errors="coerce")
#     df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
#     df.columns = df.columns.str.lower()

#     return df


# def rename_with_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
#     """Prefix all columns except date."""
#     out = df.copy()
#     for col in list(out.columns):
#         if col != "date":
#             out.rename(columns={col: f"{prefix}_{col}"}, inplace=True)
#     return out


# def merge_all():
#     # Load all indicator files
#     axjo = rename_with_prefix(load_latest("AXJO"), "axjo")
#     spy  = rename_with_prefix(load_latest("SPY"), "spy")
#     ftse = rename_with_prefix(load_latest("FTSE"), "ftse")
#     n225 = rename_with_prefix(load_latest("N225"), "n225")

#     # Merge on date (outer join keeps all date ranges)
#     merged = axjo.merge(spy, on="date", how="outer")
#     merged = merged.merge(ftse, on="date", how="outer")
#     merged = merged.merge(n225, on="date", how="outer")

#     # Clean
#     merged = merged.sort_values("date").reset_index(drop=True)
#     merged = merged.ffill()

#     # Save final dataset
#     outpath = OUTDIR / "merged_features.parquet"
#     merged.to_parquet(outpath, index=False)

#     print(f"[OK] Merged dataset saved → {outpath}")
#     print(f"[INFO] Final shape: {merged.shape}")

#     return merged


# if __name__ == "__main__":
#     merge_all()

#!/usr/bin/env python3
"""
merge_data.py (final production version)

Handles:
- MultiIndex columns
- FX / commodity different calendars
- Duplicate Date issues
- Flattening tuple column names
- Column prefixing
- Three-stage merging (markets, commodities, fx)
"""

import os
import yaml
import pandas as pd

CONFIG_PATH = "config/config.yaml"
PROCESSED_DIR = "data/processed"
MERGED_DIR = "data/merged"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def flatten_dict_lists(d):
    out = []
    for v in d.values():
        if isinstance(v, list):
            out.extend(v)
    return out


def gather_by_group(cfg):
    """Return three lists: markets, commodities, fx."""
    markets = flatten_dict_lists(cfg["markets"]) if cfg["include"]["markets"] else []
    commodities = cfg["commodities"] if cfg["include"]["commodities"] else []
    fx = cfg["fx"] if cfg["include"]["fx"] else []
    return markets, commodities, fx


# ---------------------------------------------------------
# Cleaning utilities
# ---------------------------------------------------------

def flatten_columns(df):
    """Flatten MultiIndex or tuple columns."""
    new_cols = []

    for col in df.columns:
        if isinstance(col, tuple):
            col = "_".join([str(c) for c in col if c not in ("", None)])
        new_cols.append(col)

    df.columns = new_cols
    return df


def force_date_column(df):
    """Ensure a single Date column exists and is datetime."""
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # find all date-like columns
    date_candidates = [c for c in df.columns if str(c).lower() in ("date", "datetime")]

    if len(date_candidates) == 0:
        raise RuntimeError("No Date column found.")

    if len(date_candidates) > 1:
        keep = date_candidates[-1]
        drop = date_candidates[:-1]
        df = df.drop(columns=drop)
        df = df.rename(columns={keep: "Date"})
    else:
        df = df.rename(columns={date_candidates[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()]
    return df


def prefix(df, symbol):
    """Prefix all columns except Date."""
    rename_map = {}
    for col in df.columns:
        if col != "Date":
            rename_map[col] = f"{symbol}_{col}"
    return df.rename(columns=rename_map)


def load_and_clean_symbol(symbol):
    """Load, flatten, sanitize, prefix."""
    path = f"{PROCESSED_DIR}/{symbol}.parquet"
    if not os.path.exists(path):
        print(f"[WARN] Missing: {symbol}")
        return None

    df = pd.read_parquet(path)

    df = flatten_columns(df)
    df = force_date_column(df)
    df = prefix(df, symbol)

    return df


def merge_group(symbols):
    """Merge a group (markets, commodities, fx)."""
    merged = None
    for symbol in symbols:
        df = load_and_clean_symbol(symbol)
        if df is None:
            continue

        if merged is None:
            merged = df
        else:
            merged = pd.merge(
                merged, df,
                how="outer",
                on="Date"
            )

    if merged is not None:
        merged = merged.sort_values("Date").ffill()

    return merged


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    cfg = load_config()
    markets, commodities, fx = gather_by_group(cfg)

    os.makedirs(MERGED_DIR, exist_ok=True)

    print("[INFO] Merging markets...")
    df_markets = merge_group(markets)

    print("[INFO] Merging commodities...")
    df_commodities = merge_group(commodities)

    print("[INFO] Merging FX...")
    df_fx = merge_group(fx)

    # Stage 3 — merge category-level datasets
    full = df_markets

    if df_commodities is not None:
        full = pd.merge(full, df_commodities, how="outer", on="Date")

    if df_fx is not None:
        full = pd.merge(full, df_fx, how="outer", on="Date")

    full = full.sort_values("Date").ffill()

    out = f"{MERGED_DIR}/merged_features.parquet"
    full.to_parquet(out)

    print(f"[OK] Final merged dataset saved to: {out}")


if __name__ == "__main__":
    main()
# python3 scripts/merge_data.py
