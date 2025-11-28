#!/usr/bin/env python3
"""
extract_abs_gdp_tot_filtered_v3.py

Final, production-grade extractor for ABS GDP & ToT PCA factors (Option C).

Design:
- Use ONLY true business-cycle aggregates for GDP:
    Components (prefixes):
        GNI, GPM, NDP, NNDI, RDI, GVA, SAV
    Measures:
        M1, M3 only
    Suffix:
        Prefer *_10_Q, *_20_Q, allow *_M1_Q, *_M3_Q
- Hard-coded deny-list to exclude:
    HRW, PHW, PCA, MKT, LAB, RULC, HSR, DFD, _30_Q, M5_, M6_
- ToT:
    All AUS_GDP_* files containing "TTR" (Terms of Trade)

Input:
    data/processed_abs/abs_all_series/AUS_GDP_*.parquet  (canonical ABS extracts)

Output:
    data/processed_abs/AUS_GDP.parquet
    data/processed_abs/AUS_TOT.parquet

Schema:
    date, value, series, region, source
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ABS_ALL_DIR = Path("data/processed_abs/abs_all_series")
OUT_DIR = Path("data/processed_abs")
OUT_GDP = OUT_DIR / "AUS_GDP.parquet"
OUT_TOT = OUT_DIR / "AUS_TOT.parquet"


# =============================================================================
# ALLOW-LIST COMPONENTS (Option C)
# =============================================================================
ALLOWED_COMPONENT_PREFIXES = [
    "GNI",   # Gross National Income
    "GPM",   # GDP, market prices (headline)
    "NDP",   # Net Domestic Product
    "NNDI",  # Net National Disposable Income
    "RDI",   # Real Disposable Income
    "GVA",   # Gross Value Added
    "SAV",   # Saving / saving ratio
]

# =============================================================================
# DENY-LIST SUBSTRINGS
# =============================================================================
DENY_SUBSTRINGS = [
    "HRW",     # hours worked variants
    "PHW",     # production-hours worked decomposition
    "PCA",     # ABS PCA subcomponents
    "MKT",     # market-only decompositions
    "LAB",     # labour productivity
    "RULC",    # real unit labour costs
    "HSR",     # smoothing residuals
    "DFD",     # domestic final demand, not GDP
    "_30_Q",   # truncated series ending early
    "M5_",     # measures M5/M6 we don't want
    "M6_",
]


# =============================================================================
# HELPERS
# =============================================================================

def load_series(path: Path) -> pd.Series:
    """Load a canonical ABS series (date, value, ...) as a pandas Series."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].astype(float)


def has_valid_measure_token(parts) -> bool:
    """
    True if any token in the filename parts looks like M1* or M3*.
    E.g., M1, M1_10, M1_20, M3, M3_10, M3_20.
    """
    for p in parts:
        if p.startswith("M1") or p.startswith("M3"):
            return True
    return False


def has_valid_quarter_suffix(stem: str) -> bool:
    """
    Accept only filenames that look like proper quarterly M1/M3 series.
    Allowed patterns:
        *_10_Q, *_20_Q, *_M1_Q, *_M3_Q
    """
    stem = stem.upper()
    if stem.endswith("_10_Q"):
        return True
    if stem.endswith("_20_Q"):
        return True
    if stem.endswith("_M1_Q"):
        return True
    if stem.endswith("_M3_Q"):
        return True
    return False


def is_valid_gdp_file(fp: Path) -> bool:
    """
    True if filename is a valid GDP component for PCA (Option C).

    Conditions:
        - Starts with AUS_GDP_
        - Does NOT contain TTR (ToT handled separately)
        - Does NOT contain any deny-list substring
        - Component (3rd token) starts with one of ALLOWED_COMPONENT_PREFIXES
        - Has a measure token M1* or M3*
        - Has a valid quarterly suffix
    """
    stem = fp.stem.upper()

    # Only AUS_GDP_* files
    if not stem.startswith("AUS_GDP_"):
        return False

    # Exclude ToT – those go into AUS_TOT
    if "TTR" in stem:
        return False

    # Deny-list for polluted variants
    for bad in DENY_SUBSTRINGS:
        if bad in stem:
            return False

    parts = stem.split("_")
    if len(parts) < 3:
        return False

    component = parts[2]  # e.g., GPM, GNI, NDP, NNDI, RDI, GVA, SAV

    if not any(component.startswith(pref) for pref in ALLOWED_COMPONENT_PREFIXES):
        return False

    if not has_valid_measure_token(parts):
        return False

    if not has_valid_quarter_suffix(stem):
        return False

    return True


def pca_first_factor(df_wide: pd.DataFrame) -> pd.Series:
    """
    Compute the first PCA factor (PC1) from a wide DataFrame:

        index: date (quarterly)
        columns: different GDP / ToT series

    Steps:
        - Drop columns with < 20 non-null obs or zero std
        - Drop any rows with NaNs (overlapping region only)
        - Standardize columns (z-score)
        - PCA via covariance eigendecomposition
        - Take eigenvector with largest eigenvalue (PC1)
        - Return normalized factor as pd.Series (mean 0, std 1)
    """
    # Clean columns
    cleaned = {}
    for col in df_wide.columns:
        s = df_wide[col].dropna()
        if len(s) >= 20 and s.std() > 0 and not np.isnan(s.std()):
            cleaned[col] = df_wide[col]

    if len(cleaned) < 2:
        raise RuntimeError("Need at least 2 valid series for PCA.")

    df = pd.DataFrame(cleaned).dropna(how="any")
    if df.empty:
        raise RuntimeError("No overlapping region for PCA.")

    # Standardize
    df_std = (df - df.mean()) / df.std()

    cov = np.cov(df_std.values, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(-eigvals)
    eigvecs = eigvecs[:, idx]
    pc1 = eigvecs[:, 0]

    factor_vals = df_std.values @ pc1
    factor = pd.Series(factor_vals, index=df_std.index, name="value")

    # Normalize factor
    factor = (factor - factor.mean()) / factor.std()

    return factor


def factor_to_daily(df_factor: pd.DataFrame, series_name: str, source: str) -> pd.DataFrame:
    """
    Convert quarterly factor to daily canonical series with forward-fill.
    """
    df = df_factor.copy()
    df = df.set_index("date").sort_index()
    df = df.asfreq("D")
    df["value"] = df["value"].ffill()

    df["series"] = series_name
    df["region"] = "AUS"
    df["source"] = source

    df = df.reset_index()
    return df[["date", "value", "series", "region", "source"]]


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("[GDP FILTER V3] Starting filtered PCA extraction for AUS_GDP and AUS_TOT...")

    if not ABS_ALL_DIR.exists():
        raise FileNotFoundError(f"ABS directory missing: {ABS_ALL_DIR}")

    all_files = sorted(ABS_ALL_DIR.glob("AUS_GDP_*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No AUS_GDP_* files in {ABS_ALL_DIR}")

    # Partition GDP vs ToT
    gdp_candidates = [fp for fp in all_files if is_valid_gdp_file(fp)]
    tot_candidates = [fp for fp in all_files if "TTR" in fp.stem.upper()]

    print(f"[FILTER] Valid GDP PCA components (Option C): {len(gdp_candidates)}")
    for fp in gdp_candidates:
        print(f"    - {fp.name}")

    print(f"[FILTER] Valid ToT PCA components: {len(tot_candidates)}")
    for fp in tot_candidates:
        print(f"    - {fp.name}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # GDP PCA
    # ---------------------------------------------------------------------
    if gdp_candidates:
        gdp_dict = {fp.stem: load_series(fp) for fp in gdp_candidates}
        df_gdp = pd.concat(gdp_dict, axis=1)

        gdp_factor = pca_first_factor(df_gdp)
        df_gdp_factor = gdp_factor.reset_index().rename(columns={"index": "date"})
        df_gdp_daily = factor_to_daily(df_gdp_factor, "AUS_GDP", "ABS_PCA_GDP")

        df_gdp_daily.to_parquet(OUT_GDP, index=False)
        print(f"[SAVE] Clean AUS_GDP → {OUT_GDP}")
    else:
        print("[WARN] No valid GDP candidates passed the filters.")

    # ---------------------------------------------------------------------
    # ToT PCA
    # ---------------------------------------------------------------------
    if tot_candidates:
        tot_dict = {fp.stem: load_series(fp) for fp in tot_candidates}
        df_tot = pd.concat(tot_dict, axis=1)

        tot_factor = pca_first_factor(df_tot)
        df_tot_factor = tot_factor.reset_index().rename(columns={"index": "date"})
        df_tot_daily = factor_to_daily(df_tot_factor, "AUS_TOT", "ABS_PCA_TOT")

        df_tot_daily.to_parquet(OUT_TOT, index=False)
        print(f"[SAVE] Clean AUS_TOT → {OUT_TOT}")
    else:
        print("[WARN] No ToT (TTR) candidates found.")

    print("[DONE] Filtered GDP/ToT PCA extraction (v3) complete.")


if __name__ == "__main__":
    main()
