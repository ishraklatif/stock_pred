#!/usr/bin/env python3
"""
inspect_pca_factors_filtered.py

Deep inspection of FILTERED PCA GDP/ToT extraction.
This script uses the SAME filtering logic as extract_abs_gdp_tot_filtered_v3.py.

Outputs:
    data/inspection/abs_pca_inspection_filtered.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sys
import io


# =============================================================================
# PATHS
# =============================================================================

ABS_ALL_DIR = Path("data/processed_abs/abs_all_series")
GDP_PATH = Path("data/processed_abs/AUS_GDP.parquet")
TOT_PATH = Path("data/processed_abs/AUS_TOT.parquet")

OUT_DIR = Path("data/inspection")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "abs_pca_inspection_filtered.txt"


# =============================================================================
# PREFIX ALLOW-LIST (Option C)
# =============================================================================

ALLOWED_COMPONENT_PREFIXES = [
    "GNI",
    "GPM",
    "NDP",
    "NNDI",
    "RDI",
    "GVA",
    "SAV",
]

DENY_SUBSTRINGS = [
    "HRW",
    "PHW",
    "PCA",
    "MKT",
    "LAB",
    "RULC",
    "HSR",
    "DFD",
    "_30_Q",
    "M5_",
    "M6_",
]


def has_valid_measure(parts):
    for p in parts:
        if p.startswith("M1") or p.startswith("M3"):
            return True
    return False


def has_valid_quarter_suffix(stem):
    return (
        stem.endswith("_10_Q")
        or stem.endswith("_20_Q")
        or stem.endswith("_M1_Q")
        or stem.endswith("_M3_Q")
    )


def is_valid_gdp_file(fp: Path) -> (bool, str):
    """
    Returns (valid: bool, reason: str)
    """
    s = fp.stem.upper()
    parts = s.split("_")

    # Only AUS_GDP_* files
    if not s.startswith("AUS_GDP_"):
        return False, "NOT_AUS_GDP"

    # Exclude ToT
    if "TTR" in s:
        return False, "IS_TOT"

    # Deny-list
    for bad in DENY_SUBSTRINGS:
        if bad in s:
            return False, f"DENIED:{bad}"

    # Component prefix
    if len(parts) < 3:
        return False, "BAD_FORMAT"

    component = parts[2]
    if not any(component.startswith(pref) for pref in ALLOWED_COMPONENT_PREFIXES):
        return False, f"BAD_PREFIX:{component}"

    # Measure token
    if not has_valid_measure(parts):
        return False, "NO_M1_M3"

    # Quarter suffix
    if not has_valid_quarter_suffix(s):
        return False, "BAD_SUFFIX"

    return True, "OK"


# =============================================================================
# HELPERS
# =============================================================================

def load_series(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].astype(float)


def pca_first_factor(df_wide):
    cleaned = {}
    for col in df_wide.columns:
        s = df_wide[col].dropna()
        if len(s) >= 20 and s.std() > 0:
            cleaned[col] = df_wide[col]
    df = pd.DataFrame(cleaned).dropna(how="any")
    df_std = (df - df.mean()) / df.std()

    cov = np.cov(df_std.values, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(-eigvals)
    eigvecs = eigvecs[:, idx]
    pc1 = eigvecs[:, 0]

    f = df_std.values @ pc1
    s = pd.Series(f, index=df_std.index, name="value")
    s = (s - s.mean()) / s.std()
    return s, eigvals[idx], eigvecs[:, idx]


def to_quarterly(df):
    return df.set_index("date").resample("QE").first()


# =============================================================================
# MAIN
# =============================================================================

def main():
    buffer = io.StringIO()
    sys.stdout = buffer

    print("=" * 90)
    print("[INSPECT] FILTERED PCA GDP/ToT inspection")
    print("=" * 90)

    all_files = sorted(ABS_ALL_DIR.glob("AUS_GDP_*.parquet"))

    included = []
    excluded = []

    for fp in all_files:
        valid, reason = is_valid_gdp_file(fp)
        if valid:
            included.append(fp)
        else:
            excluded.append((fp, reason))

    print("\n[INCLUDED FILES] (filtered GDP inputs)")
    for fp in included:
        print("  ✔", fp.name)

    print("\n[EXCLUDED FILES] (and reasons)")
    for fp, reason in excluded:
        print(f"  ✖ {fp.name:45s} → {reason}")

    print(f"\n[SUMMARY] Included: {len(included)} | Excluded: {len(excluded)}")

    # =====================================================================
    # PCA on filtered GDP
    # =====================================================================
    print("\n[INSPECT] PCA on filtered GDP set...")
    gdp_dict = {fp.stem: load_series(fp) for fp in included}
    df_gdp = pd.concat(gdp_dict, axis=1)

    print("[SHAPE] Filtered GDP matrix:", df_gdp.shape)
    print("[MISSING] Nulls per column:")
    print(df_gdp.isna().sum())

    factor, eigvals, eigvecs = pca_first_factor(df_gdp)

    print("\n[EIGENVALUES]")
    total = eigvals.sum()
    for i, v in enumerate(eigvals[:10]):
        print(f"  PC{i+1}: {v:.4f} ({v/total*100:.2f}%)")

    print("\n[PC1 LOADINGS]")
    cols = df_gdp.dropna().columns
    pc1_vec = eigvecs[:, 0]
    for col, w in zip(cols, pc1_vec):
        print(f"  {col:45s} {w:+.4f}")

    # =====================================================================
    # Inspect canonical daily AUS_GDP
    # =====================================================================
    if GDP_PATH.exists():
        print("\n[CANONICAL GDP]")
        df = pd.read_parquet(GDP_PATH)
        print(df.head(10))
        print(df.tail(10))
        print("\n[QUARTERLY GDP]")
        print(to_quarterly(df).head(12))
        print(to_quarterly(df).tail(12))
    else:
        print("\n[WARN] AUS_GDP.parquet missing")

    # =====================================================================
    # Inspect canonical daily AUS_TOT
    # =====================================================================
    if TOT_PATH.exists():
        print("\n[CANONICAL ToT]")
        df = pd.read_parquet(TOT_PATH)
        print(df.head(10))
        print(df.tail(10))
        print("\n[QUARTERLY ToT]")
        print(to_quarterly(df).head(12))
        print(to_quarterly(df).tail(12))
    else:
        print("\n[WARN] AUS_TOT.parquet missing")

    # =====================================================================
    # SAVE REPORT
    # =====================================================================
    sys.stdout = sys.__stdout__
    with OUT_FILE.open("w") as f:
        f.write(buffer.getvalue())

    print(f"[OK] Filtered inspection saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
