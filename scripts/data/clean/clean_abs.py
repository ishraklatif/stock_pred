#!/usr/bin/env python3
"""
clean_abs_auto.py — FINAL VERSION
---------------------------------

Auto-selects and cleans ABS GDP + ToT + CPI + LF + WPI + HPI.

Enhancements:
    ✔ Full quarterly parsing support ("2003-Q1")
    ✔ Full monthly parsing support
    ✔ Relaxed validation for sparse GDP/ToT tables
    ✔ Robust frequency scoring with tolerance
    ✔ Canonical output filenames:
          AUS_GDP.parquet
          AUS_TOT.parquet
    ✔ All other ABS sets kept (CPI/LF/WPI/HPI)
    ✔ Everything else archived → raw_abs/skipped/
"""

import os
import shutil
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw_abs")
OUT_DIR = Path("data/processed_abs")
SKIPPED_DIR = RAW_DIR / "skipped"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SKIPPED_DIR, exist_ok=True)


# =============================================================================
# UTILITIES
# =============================================================================
def parse_abs_date(df):
    """
    Robust date parser for ABS data.
    Supports:
        - YYYY-MM-DD
        - YYYY-MM
        - YYYY-QX (quarterly)
    """
    df = df.copy()

    # Initial parse (monthly / daily)
    df["date"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")

    # Quarterly format patch
    qmask = df["date"].isna() & df["TIME_PERIOD"].astype(str).str.contains("Q")
    if qmask.any():
        try:
            df.loc[qmask, "date"] = pd.PeriodIndex(
                df.loc[qmask, "TIME_PERIOD"], freq="Q"
            ).to_timestamp()
        except Exception:
            pass

    return df


def is_valid_abs(df):
    """
    GDP/ToT tables often sparse. We only require:
        - TIME_PERIOD exists
        - OBS_VALUE exists
        - at least 5 numeric values
    """
    if df is None:
        return False
    if "TIME_PERIOD" not in df.columns:
        return False
    if "OBS_VALUE" not in df.columns:
        return False

    vals = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    return vals.notna().sum() >= 5


def classify_abs_file(fname):
    f = fname.lower()

    if "gdp" in f:
        return "GDP"
    if "ttr" in f or "tot" in f:
        return "TOT"
    if "cpi" in f:
        return "CPI"
    if "lf" in f:
        return "LF"
    if "wpi" in f:
        return "WPI"
    if "hpi" in f or "rppi" in f:
        return "HPI"

    return "OTHER"


def load_abs(path):
    try:
        return pd.read_parquet(path)
    except:
        return None


def compute_quarter_score(df):
    """
    Compute how close frequency is to quarterly.
    Returns None if irregular beyond tolerance.
    """

    df = parse_abs_date(df)
    df = df.dropna(subset=["date"]).sort_values("date")

    if df.empty:
        return None

    diffs = df["date"].diff().dt.days.dropna()
    if diffs.empty:
        return None

    mean_gap = diffs.mean()

    # Tolerance: quarterly ≈ 91 days ± 200 days (GDP can be messy)
    if abs(mean_gap - 91) > 200:
        return None

    # Lower is better
    return abs(mean_gap - 91)


def compute_coverage(df):
    df = parse_abs_date(df)
    df = df.dropna(subset=["date"]).sort_values("date")
    if df.empty:
        return 0
    return (df["date"].iloc[-1] - df["date"].iloc[0]).days


def compute_variance(df):
    vals = pd.to_numeric(df["OBS_VALUE"], errors="coerce").dropna()
    return vals.var() if len(vals) > 5 else 0


def pick_best(files):
    """
    Picks best GDP/ToT file by:
        1. Quarterly match
        2. Longest coverage
        3. Highest variance
    """
    scored = []

    for path in files:
        df = load_abs(path)
        if df is None or not is_valid_abs(df):
            continue

        qscore = compute_quarter_score(df)
        if qscore is None:
            continue

        scored.append({
            "file": path,
            "qscore": qscore,
            "coverage": compute_coverage(df),
            "variance": compute_variance(df),
        })

    if not scored:
        return None

    # Sort: quarter closeness → coverage → variance
    best = sorted(
        scored,
        key=lambda x: (x["qscore"], -x["coverage"], -x["variance"])
    )[0]

    return best["file"]


def clean_abs(df, canonical_name):
    """Standard ABS cleaner."""
    df = df.copy()
    df = parse_abs_date(df)
    df["value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date")

    df["series"] = canonical_name
    df["region"] = "AUS"
    df["source"] = "ABS"

    return df[["date", "value", "series", "region", "source"]]


# =============================================================================
# MAIN LOGIC
# =============================================================================
def main():
    print("\n" + "="*80)
    print("[AUTO] ABS Cleaner with improved GDP/ToT detection + canonical renaming")
    print("="*80)

    raw_files = [RAW_DIR / f for f in os.listdir(RAW_DIR) if f.endswith(".parquet")]

    groups = {"GDP": [], "TOT": [], "CPI": [], "LF": [], "WPI": [], "HPI": [], "OTHER": []}

    for filepath in raw_files:
        label = classify_abs_file(filepath.name)
        groups[label].append(filepath)

    # ------------------------------------------------------------------
    # AUTO-SELECT GDP & TOT
    # ------------------------------------------------------------------
    best_gdp = pick_best(groups["GDP"])
    best_tot = pick_best(groups["TOT"])

    print(f"\n[SELECT] GDP → {best_gdp.name if best_gdp else 'NONE'}")
    print(f"[SELECT] ToT → {best_tot.name if best_tot else 'NONE'}\n")

    # Files to keep
    to_clean = {}

    if best_gdp:
        to_clean["AUS_GDP"] = best_gdp
    if best_tot:
        to_clean["AUS_TOT"] = best_tot

    # Keep CPI, LF, WPI, HPI raw names
    for grp in ["CPI", "LF", "WPI", "HPI"]:
        for f in groups[grp]:
            to_clean[f.stem] = f

    # Everything else → archive
    for f in raw_files:
        if f not in to_clean.values():
            dst = SKIPPED_DIR / f.name
            shutil.move(f, dst)
            print(f"[ARCHIVE] {f.name}")

    # ------------------------------------------------------------------
    # CLEAN + WRITE OUTPUTS
    # ------------------------------------------------------------------
    for canonical_name, fpath in to_clean.items():
        df = load_abs(fpath)

        if df is None or not is_valid_abs(df):
            print(f"[WARN] Invalid ABS file, skipping: {fpath.name}")
            continue

        cleaned = clean_abs(df, canonical_name)

        out_path = OUT_DIR / f"{canonical_name}.parquet"
        cleaned.to_parquet(out_path, index=False)
        print(f"[OK] Cleaned → {out_path}")

    print("\n[COMPLETE] ABS auto-clean finished.\n")


if __name__ == "__main__":
    main()

