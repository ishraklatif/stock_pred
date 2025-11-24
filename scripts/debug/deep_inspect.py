#!/usr/bin/env python3
"""
deep_inspect.py

Deep data inspection for TFT multiseries dataset (StockPred).

Checks:
- Basic structure (shape, dtypes)
- NaNs and Infs
- Duplicates per (series, Date) and (series, time_idx)
- time_idx continuity per series
- Per-series coverage (min/max Date & time_idx, length)
- Zero-variance & low-uniqueness numeric features
- Distribution stats and extreme quantiles
- Column consistency across train/val/test
- Time split sanity (overlap/gap between splits)
- Heuristic target leakage check via correlation with future close

Outputs multiple CSV/TXT reports into diagnostics/.
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

from scipy.stats import pearsonr

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DATA_DIR = "data/tft_ready_multiseries"
TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
VAL_PATH = os.path.join(DATA_DIR, "val.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test.parquet")

OUTPUT_DIR = "diagnostics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "close"
TIME_IDX_COL = "time_idx"
SERIES_COL = "series"
DATE_COL = "Date"


# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------
def save_text(name: str, text: str):
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, "w") as f:
        f.write(text)
    print(f"[INFO] Saved text report → {path}")


def load_datasets():
    dfs = {}
    for split, path in [("train", TRAIN_PATH),
                        ("val", VAL_PATH),
                        ("test", TEST_PATH)]:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Enforce Date as datetime if present
            if DATE_COL in df.columns and not np.issubdtype(df[DATE_COL].dtype, np.datetime64):
                df[DATE_COL] = pd.to_datetime(df[DATE_COL])
            dfs[split] = df
            print(f"[LOAD] {split}: {len(df):,} rows, {df.shape[1]} cols from {path}")
        else:
            print(f"[WARN] Missing file for {split}: {path}")
    return dfs


# -------------------------------------------------------------------
# BASIC STRUCTURE
# -------------------------------------------------------------------
def check_basic_info(dfs):
    lines = []
    for name, df in dfs.items():
        lines.append(f"=== {name.upper()} ===")
        lines.append(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
        lines.append("Column dtypes:")
        lines.append(str(df.dtypes))
        lines.append("")
    text = "\n".join(lines)
    save_text("basic_info.txt", text)


# -------------------------------------------------------------------
# NANS & INFS
# -------------------------------------------------------------------
def check_nans_infs(dfs):
    lines = []
    for name, df in dfs.items():
        n_nan = df.isna().sum().sum()
        n_posinf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        n_neginf = np.isneginf(df.select_dtypes(include=[np.number])).sum().sum()

        lines.append(f"=== {name.upper()} ===")
        lines.append(f"Total NaNs: {n_nan}")
        lines.append(f"Total +inf: {n_posinf}")
        lines.append(f"Total -inf: {n_neginf}")
        lines.append("")

        # Per-column NaNs
        nan_counts = df.isna().sum()
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        nan_counts[nan_counts > 0].sort_values(ascending=False).to_csv(
            os.path.join(OUTPUT_DIR, f"{name}_nan_counts.csv")
        )
        inf_counts[inf_counts > 0].sort_values(ascending=False).to_csv(
            os.path.join(OUTPUT_DIR, f"{name}_inf_counts.csv")
        )
    text = "\n".join(lines)
    save_text("nan_inf_summary.txt", text)


# -------------------------------------------------------------------
# DUPLICATES
# -------------------------------------------------------------------
def check_duplicates(dfs):
    lines = []
    for name, df in dfs.items():
        lines.append(f"=== {name.upper()} ===")
        if {SERIES_COL, DATE_COL}.issubset(df.columns):
            dup_sd = df.duplicated(subset=[SERIES_COL, DATE_COL]).sum()
            lines.append(f"Duplicates (series, Date): {dup_sd}")
        else:
            lines.append("(series, Date) not fully available")

        if {SERIES_COL, TIME_IDX_COL}.issubset(df.columns):
            dup_st = df.duplicated(subset=[SERIES_COL, TIME_IDX_COL]).sum()
            lines.append(f"Duplicates (series, time_idx): {dup_st}")
        else:
            lines.append("(series, time_idx) not fully available")

        lines.append("")
    text = "\n".join(lines)
    save_text("duplicates.txt", text)


# -------------------------------------------------------------------
# TIME_IDX CONTINUITY PER SERIES
# -------------------------------------------------------------------
def check_time_idx_continuity(dfs):
    lines = []
    for name, df in dfs.items():
        if {SERIES_COL, TIME_IDX_COL}.issubset(df.columns):
            broken_series = []
            details = []

            for s, g in df.groupby(SERIES_COL):
                g = g.sort_values(TIME_IDX_COL)
                diffs = g[TIME_IDX_COL].diff().dropna()
                if not (diffs == 1).all():
                    broken_series.append(s)
                    details.append({
                        SERIES_COL: s,
                        "min_diff": diffs.min(),
                        "max_diff": diffs.max(),
                        "n_rows": len(g),
                    })

            lines.append(f"=== {name.upper()} ===")
            lines.append(f"Series with non-contiguous {TIME_IDX_COL}: {len(broken_series)}")
            if details:
                cont_df = pd.DataFrame(details)
                cont_df.to_csv(os.path.join(OUTPUT_DIR, f"{name}_time_idx_continuity_issues.csv"),
                               index=False)
                lines.append(f"Saved details → {name}_time_idx_continuity_issues.csv")
            lines.append("")
        else:
            lines.append(f"=== {name.upper()} ===")
            lines.append("Missing series or time_idx columns; skipping continuity check.")
            lines.append("")
    text = "\n".join(lines)
    save_text("time_idx_continuity.txt", text)


# -------------------------------------------------------------------
# PER-SERIES COVERAGE
# -------------------------------------------------------------------
def check_series_coverage(dfs):
    for name, df in dfs.items():
        if {SERIES_COL, TIME_IDX_COL, DATE_COL}.issubset(df.columns):
            cov = df.groupby(SERIES_COL).agg(
                n_rows=(TIME_IDX_COL, "count"),
                time_idx_min=(TIME_IDX_COL, "min"),
                time_idx_max=(TIME_IDX_COL, "max"),
                date_min=(DATE_COL, "min"),
                date_max=(DATE_COL, "max"),
            ).reset_index()
            cov.to_csv(os.path.join(OUTPUT_DIR, f"{name}_series_coverage.csv"), index=False)
            print(f"[INFO] Saved {name} series coverage → {OUTPUT_DIR}/{name}_series_coverage.csv")


# -------------------------------------------------------------------
# FEATURE VARIABILITY (ZERO VAR, LOW UNIQUE)
# -------------------------------------------------------------------
def check_feature_variability(dfs):
    for name, df in dfs.items():
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            continue

        stats = []

        for col in num_cols:
            series = df[col]
            var = series.var()
            uniq = series.nunique(dropna=True)
            stats.append({
                "column": col,
                "variance": var,
                "n_unique": uniq,
            })

        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(os.path.join(OUTPUT_DIR, f"{name}_numeric_variability.csv"), index=False)

        zero_var = stats_df[stats_df["variance"] == 0]
        low_unique = stats_df[stats_df["n_unique"] <= 2]

        zero_var.to_csv(os.path.join(OUTPUT_DIR, f"{name}_zero_variance_features.csv"), index=False)
        low_unique.to_csv(os.path.join(OUTPUT_DIR, f"{name}_low_unique_features.csv"), index=False)

        print(f"[INFO] {name}: {len(zero_var)} zero-variance, {len(low_unique)} low-unique features")


# -------------------------------------------------------------------
# DISTRIBUTION STATISTICS & QUANTILES
# -------------------------------------------------------------------
def check_distributions(dfs):
    for name, df in dfs.items():
        num_cols = df.select_dtypes(include=[np.number]).columns
        if num_cols.empty:
            continue

        desc = df[num_cols].describe(percentiles=[0.01, 0.05, 0.95, 0.99]).T
        desc.to_csv(os.path.join(OUTPUT_DIR, f"{name}_numeric_distribution_stats.csv"))
        print(f"[INFO] {name}: saved distribution stats for numeric features.")


# -------------------------------------------------------------------
# COLUMN CONSISTENCY ACROSS SPLITS
# -------------------------------------------------------------------
def check_column_consistency(dfs):
    col_sets = {name: set(df.columns) for name, df in dfs.items()}
    all_cols = set.union(*col_sets.values()) if col_sets else set()

    lines = ["COLUMN CONSISTENCY ACROSS SPLITS", ""]

    for name, cols in col_sets.items():
        missing = all_cols - cols
        extra = cols - (all_cols - cols)
        lines.append(f"=== {name.upper()} ===")
        lines.append(f"Total cols: {len(cols)}")
        lines.append(f"Missing columns vs union: {missing}")
        lines.append("")

    # pairwise differences
    lines.append("PAIRWISE DIFFERENCES:")
    names = list(col_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            only_a = col_sets[a] - col_sets[b]
            only_b = col_sets[b] - col_sets[a]
            lines.append(f"{a} - {b}: {only_a}")
            lines.append(f"{b} - {a}: {only_b}")
            lines.append("")

    save_text("column_consistency.txt", "\n".join(lines))


# -------------------------------------------------------------------
# TIME SPLIT SANITY (TRAIN / VAL / TEST)
# -------------------------------------------------------------------
def check_time_splits(dfs):
    lines = ["TIME SPLIT CHECK", ""]

    for name, df in dfs.items():
        if TIME_IDX_COL in df.columns and DATE_COL in df.columns:
            ti_min = df[TIME_IDX_COL].min()
            ti_max = df[TIME_IDX_COL].max()
            d_min = df[DATE_COL].min()
            d_max = df[DATE_COL].max()
            lines.append(f"=== {name.upper()} ===")
            lines.append(f"time_idx: [{ti_min}, {ti_max}]")
            lines.append(f"Date:     [{d_min}, {d_max}]")
            lines.append("")
        else:
            lines.append(f"=== {name.upper()} ===")
            lines.append("Missing time_idx or Date")
            lines.append("")

    # overlap/gap checks if all splits exist
    if all(k in dfs for k in ("train", "val", "test")):
        tr, va, te = dfs["train"], dfs["val"], dfs["test"]
        if TIME_IDX_COL in tr.columns and TIME_IDX_COL in va.columns and TIME_IDX_COL in te.columns:
            tr_max = tr[TIME_IDX_COL].max()
            va_min = va[TIME_IDX_COL].min()
            va_max = va[TIME_IDX_COL].max()
            te_min = te[TIME_IDX_COL].min()

            lines.append("SPLIT RELATIONSHIPS (time_idx):")
            lines.append(f"train max = {tr_max}, val min = {va_min}")
            lines.append(f"val   max = {va_max}, test min = {te_min}")
            lines.append(f"train→val gap = {va_min - tr_max}")
            lines.append(f"val→test gap  = {te_min - va_max}")
            lines.append("Expected: gaps of 1 if strictly adjacent with no overlap.")
            lines.append("")

    save_text("time_splits.txt", "\n".join(lines))


# -------------------------------------------------------------------
# HEURISTIC TARGET LEAKAGE CHECK
#   - Correlation between feature(t) and close(t+1)
#   - Sample to keep runtime reasonable
# -------------------------------------------------------------------
def check_target_leakage(dfs, sample_size=20000):
    if "train" not in dfs:
        print("[WARN] No train split → skipping target leakage check.")
        return
    df = dfs["train"].copy()

    required = {SERIES_COL, TIME_IDX_COL, TARGET_COL}
    if not required.issubset(df.columns):
        print("[WARN] Missing series/time_idx/target → skipping target leakage check.")
        return

    # Sort properly
    df = df.sort_values([SERIES_COL, TIME_IDX_COL])

    # Sample for speed
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).sort_values([SERIES_COL, TIME_IDX_COL])

    # Compute future target
    df["future_close"] = df.groupby(SERIES_COL)[TARGET_COL].shift(-1)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove direct target and future target from feature list
    for drop_col in [TARGET_COL, "future_close", TIME_IDX_COL]:
        if drop_col in num_cols:
            num_cols.remove(drop_col)

    results = []
    for col in num_cols:
        x = df[col].values
        y = df["future_close"].values

        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 100:
            continue

        try:
            r, _ = pearsonr(x[mask], y[mask])
        except Exception:
            continue

        results.append({"feature": col, "corr_with_future_close": r})

    if results:
        res_df = pd.DataFrame(results).sort_values("corr_with_future_close", ascending=False)
        res_df.to_csv(os.path.join(OUTPUT_DIR, "train_feature_future_close_corr.csv"), index=False)
        print("[INFO] Saved feature vs future_close correlations → train_feature_future_close_corr.csv")
    else:
        print("[INFO] No valid correlations computed for target leakage check.")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    dfs = load_datasets()
    if not dfs:
        print("[FATAL] No datasets loaded. Check paths.")
        return

    print("\n[STEP] Basic info")
    check_basic_info(dfs)

    print("\n[STEP] NaNs & Infs")
    check_nans_infs(dfs)

    print("\n[STEP] Duplicates")
    check_duplicates(dfs)

    print("\n[STEP] time_idx continuity")
    check_time_idx_continuity(dfs)

    print("\n[STEP] Per-series coverage")
    check_series_coverage(dfs)

    print("\n[STEP] Feature variability")
    check_feature_variability(dfs)

    print("\n[STEP] Distribution stats")
    check_distributions(dfs)

    print("\n[STEP] Column consistency")
    check_column_consistency(dfs)

    print("\n[STEP] Time splits")
    check_time_splits(dfs)

    print("\n[STEP] Target leakage heuristic")
    check_target_leakage(dfs)

    print("\n[INFO] Deep inspection complete. See diagnostics/ folder.")


if __name__ == "__main__":
    main()
