#!/usr/bin/env python3
"""
inspect_pca_factors.py

Deep inspection of PCA-based AUS_GDP / AUS_TOT factors.
Writes a full detailed report to:

    data/inspection/abs_pca_inspection.txt

Paths are defined directly without ROOT detection.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import sys
import io


# ======================================================================
# PATH DEFINITIONS (your requirement)
# ======================================================================

ABS_ALL_DIR = Path("data/processed_abs/abs_all_series")
PCA_GDP_PATH = Path("data/processed_abs/AUS_GDP.parquet")
PCA_TOT_PATH = Path("data/processed_abs/AUS_TOT.parquet")

OUT_DIR = Path("data/inspection")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "abs_pca_inspection.txt"


# ======================================================================
# HELPERS
# ======================================================================

def load_series(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].astype(float)


def summarize(path: Path) -> dict:
    s = load_series(path)
    if s.empty:
        return dict(n=0, dmin=None, dmax=None)
    return dict(
        n=s.shape[0],
        dmin=s.index.min().date(),
        dmax=s.index.max().date(),
    )


def to_quarterly(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_daily.copy()
    return df.set_index("date").resample("QE").first()


def save_and_exit(buffer):
    sys.stdout = sys.__stdout__
    with OUT_FILE.open("w") as f:
        f.write(buffer.getvalue())
    print(f"[OK] Inspection saved to {OUT_FILE}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    # Capture all printed output
    buffer = io.StringIO()
    sys.stdout = buffer

    print("=" * 90)
    print("[INSPECT] Deep PCA GDP/ToT inspection")
    print("=" * 90)

    # --------------------------------------------------------
    # Load ABS files
    # --------------------------------------------------------
    if not ABS_ALL_DIR.exists():
        print("[ERROR] Directory not found:", ABS_ALL_DIR)
        return save_and_exit(buffer)

    files = sorted(ABS_ALL_DIR.glob("AUS_GDP_*.parquet"))
    if not files:
        print("[ERROR] No AUS_GDP_* files found.")
        return save_and_exit(buffer)

    # Partition
    gdp_files = [fp for fp in files if "TTR" not in fp.stem.upper()]
    tot_files = [fp for fp in files if "TTR" in fp.stem.upper()]

    print("\n[INSPECT] GDP candidate files:")
    for fp in gdp_files:
        s = summarize(fp)
        print(f"  {fp.name:40s}  n={s['n']:5d}  {s['dmin']} → {s['dmax']}")

    print("\n[INSPECT] ToT candidate files:")
    for fp in tot_files:
        s = summarize(fp)
        print(f"  {fp.name:40s}  n={s['n']:5d}  {s['dmin']} → {s['dmax']}")

    # --------------------------------------------------------
    # GDP-wide matrix
    # --------------------------------------------------------
    print("\n[INSPECT] Building GDP-wide matrix...")
    gdp_dict = {fp.stem: load_series(fp) for fp in gdp_files}
    df_gdp = pd.concat(gdp_dict, axis=1)

    print("\n[INSPECT] GDP-wide shape:", df_gdp.shape)
    print("[INSPECT] Columns:", list(df_gdp.columns))

    print("\n[INSPECT] Missing values:")
    print(df_gdp.isna().sum())

    print("\n[INSPECT] Correlation matrix:")
    print(df_gdp.corr().round(3))

    # --------------------------------------------------------
    # PCA Diagnostics
    # --------------------------------------------------------
    print("\n[INSPECT] PCA diagnostics...")

    df_clean = df_gdp.dropna(how="any")
    df_std = (df_clean - df_clean.mean()) / df_clean.std()

    cov = np.cov(df_std.values, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    explained = eigvals / eigvals.sum()

    print("\nEigenvalues (sorted):")
    for i in range(min(8, len(eigvals))):
        print(f"  PC{i+1}: {eigvals[i]:.4f} ({explained[i]*100:.2f}%)")

    print("\nLoadings for PC1:")
    pc1 = eigvecs[:, 0]
    for col, weight in zip(df_clean.columns, pc1):
        print(f"  {col:40s}  {weight:+.4f}")

    # --------------------------------------------------------
    # Inspect canonical PCA GDP
    # --------------------------------------------------------
    if PCA_GDP_PATH.exists():
        print("\n[INSPECT] Canonical PCA GDP (daily):")
        df = pd.read_parquet(PCA_GDP_PATH)
        print(df.head(10))
        print(df.tail(10))

        print("\n[INSPECT] Back-converted quarterly:")
        print(to_quarterly(df).head(12))
        print(to_quarterly(df).tail(12))
    else:
        print("\n[WARN] PCA GDP file missing:", PCA_GDP_PATH)

    # --------------------------------------------------------
    # Inspect canonical PCA ToT
    # --------------------------------------------------------
    if PCA_TOT_PATH.exists():
        print("\n[INSPECT] Canonical PCA ToT (daily):")
        df = pd.read_parquet(PCA_TOT_PATH)
        print(df.head(10))
        print(df.tail(10))

        print("\n[INSPECT] Back-converted quarterly:")
        print(to_quarterly(df).head(12))
        print(to_quarterly(df).tail(12))
    else:
        print("\n[WARN] PCA ToT file missing:", PCA_TOT_PATH)

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------
    save_and_exit(buffer)


if __name__ == "__main__":
    main()
