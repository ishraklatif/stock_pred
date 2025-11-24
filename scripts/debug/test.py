# # # # # import pandas as pd

# # # # # df = pd.read_parquet("data/cache/AXJO_20251114_ind_20251114.parquet")

# # # # # print(df.head())
# # # # # print(df.columns)
# # # # # print(df.tail())

# # # # # import pandas as pd

# # # # # df = pd.read_parquet("data/merged/merged_features.parquet")

# # # # # print(df.head())
# # # # # print(df.tail())
# # # # # print(df.info())
# # # # # print(df.columns)

# # # # # print("✅ Option 2 — Check shape + missing values")

# # # # # print("Rows:", len(df))
# # # # # print("Columns:", len(df.columns))
# # # # # print(df.isna().sum().sort_values().tail(20))

# # # # # import pandas as pd
# # # # # df = pd.read_parquet("data/enriched/macro_level2.parquet")

# # # # # return_cols = [c for c in df.columns if "return" in c.lower()]
# # # # # for c in return_cols:
# # # # #     print(c)

# # # # # print("*************************COMPANY****************************")
# # # # # df2 = pd.read_parquet("data/enriched/ANZ.AX_dataset_enriched.parquet")

# # # # # return_cols = [c for c in df2.columns if "return" in c.lower()]
# # # # # for d in return_cols:
# # # # #     print(d)



# # # # import pandas as pd
# # # # df = pd.read_parquet("data/level4_company/CBA_AX_merged.parquet")
# # # # for col in df.columns:
# # # #     if "CBA" in col: print(col)

# # # import pandas as pd
# # # from pathlib import Path

# # # # Change this to any symbol
# # # path = Path("data/raw_companies/AGL.parquet")

# # # print("======== LOADING RAW DATA ========")
# # # df = pd.read_parquet(path)
# # # print(df.head())

# # # print("\n======== COLUMN NAMES ========")
# # # print(df.columns)

# # # print("\n======== COLUMN TYPES ========")
# # # for col in df.columns:
# # #     print(f"{col}: {type(col)}")

# # # print("\n======== IS MultiIndex? ========")
# # # print(isinstance(df.columns, pd.MultiIndex))

# # # print("\n======== INDEX INFO ========")
# # # print("Index type:", type(df.index))
# # # print("Index name:", df.index.name)
# # # print(df.index[:5])

# # # print("\n======== COLUMN NAME CASE CHECK ========")
# # # print([c for c in df.columns])

# # # print("\n======== FULL INFO ========")
# # # print(df.info())

# # import pandas as pd

# # df = pd.read_parquet("data/sanitised_final/company_with_sector/AGL_clean_sanitized.parquet")
# # print(df.columns)

# import pandas as pd

# df = pd.read_parquet("data/tft_ready_multiseries/train.parquet")

# print(df["close"].isna().sum())
# print(df["close"].isna().mean())
# print(df[df["close"].isna()].head())
# print(df.groupby("series")["close"].apply(lambda x: x.isna().sum()).sort_values(ascending=False).head(20))

# import pandas as pd

# df = pd.read_parquet("data/tft_ready_multiseries/train.parquet")

# missing = df[df["sector_id"] == "Unknown"]["series"].unique()
# print("Companies without sector_id:")
# for s in missing:
#     print(" -", s)

# import pandas as pd

# df_train = pd.read_parquet("data/tft_ready_multiseries/train.parquet")


# print("\nTrain columns:")
# print(df_train.columns.tolist())

# import pandas as pd
# df = pd.read_parquet("data/tft_ready_multiseries/train.parquet")
# len([c for c in df.columns if c not in ["Date", "series", "sector_id", "time_idx", "close"]])

#!/usr/bin/env python3
"""
inspect_nans.py

Full NaN diagnostics for TFT multiseries dataset.
- Loads train/val/test parquet files
- Reports:
    ✔ total NaNs
    ✔ columns with NaNs
    ✔ NaN counts per column
    ✔ NaN start and end dates per column
    ✔ NaNs per series (ticker)
    ✔ Saves all results to CSVs

"""

import pandas as pd
import os

DATA_DIR = "data/tft_ready_multiseries"
TRAIN_PATH = f"{DATA_DIR}/train.parquet"
VAL_PATH   = f"{DATA_DIR}/val.parquet"
TEST_PATH  = f"{DATA_DIR}/test.parquet"

OUTPUT_DIR = "nan_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_datasets():
    dfs = {}
    for name, path in [
        ("train", TRAIN_PATH),
        ("val", VAL_PATH),
        ("test", TEST_PATH),
    ]:
        if os.path.exists(path):
            dfs[name] = pd.read_parquet(path)
        else:
            print(f"[WARN] Missing file: {path}")
    return dfs


# -------------------------------------------------------
# Find NaN start + end date per column
# -------------------------------------------------------
def nan_date_ranges(df, date_col="Date"):
    results = []

    for col in df.columns:
        if df[col].isna().any():
            nan_rows = df[df[col].isna()]
            start_date = nan_rows[date_col].min()
            end_date   = nan_rows[date_col].max()
            count = len(nan_rows)

            results.append((col, count, start_date, end_date))

    return pd.DataFrame(
        results,
        columns=["column", "nan_count", "nan_start", "nan_end"]
    ).sort_values("nan_count", ascending=False)


# -------------------------------------------------------
# Count NaNs per series (ticker)
# -------------------------------------------------------
def nan_by_series(df, series_col="series"):
    return df.groupby(series_col).apply(
        lambda x: x.isna().sum().sum()
    ).sort_values(ascending=False)


def check_time_idx_continuity(df):
    bad = []
    for s, g in df.groupby("series"):
        diffs = g["time_idx"].diff().dropna()
        if not (diffs == 1).all():
            bad.append(s)
    return bad

bad_series = check_time_idx_continuity(train_df)
print("Series with broken time_idx:", bad_series)


# -------------------------------------------------------
# Main inspection routine
# -------------------------------------------------------
def inspect():
    dfs = load_datasets()

    for name, df in dfs.items():
        print(f"\n===============================")
        print(f"===== {name.upper()} DATA =====")
        print(f"Rows: {len(df):,}   Cols: {df.shape[1]}")
        print("===============================\n")

        # 1. Total NaNs
        total_nans = df.isna().sum().sum()
        print(f"[{name}] Total NaNs = {total_nans:,}")

        # 2. Columns with NaNs
        nan_cols = df.columns[df.isna().any()].tolist()
        print(f"[{name}] Columns with NaNs ({len(nan_cols)}):")
        print(nan_cols)

        # 3. NaN counts per column
        nan_counts = df.isna().sum().sort_values(ascending=False)
        nan_counts_df = nan_counts.reset_index()
        nan_counts_df.columns = ["column", "nan_count"]

        nan_counts_df.to_csv(f"{OUTPUT_DIR}/{name}_nan_counts.csv", index=False)
        print(f"[{name}] Saved column NaN counts → {OUTPUT_DIR}/{name}_nan_counts.csv")

        # 4. NaN date ranges per column
        ranges_df = nan_date_ranges(df)
        ranges_df.to_csv(f"{OUTPUT_DIR}/{name}_nan_date_ranges.csv", index=False)
        print(f"[{name}] Saved NaN date ranges → {OUTPUT_DIR}/{name}_nan_date_ranges.csv")

        # 5. NaNs per series
        nan_series_df = nan_by_series(df)
        nan_series_df.to_csv(f"{OUTPUT_DIR}/{name}_nan_by_series.csv")
        print(f"[{name}] Saved NaNs by series → {OUTPUT_DIR}/{name}_nan_by_series.csv")

        print("\nDone.\n")


if __name__ == "__main__":
    inspect()



