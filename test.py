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

import pandas as pd

df = pd.read_parquet("data/tft_ready_multiseries/train.parquet")

missing = df[df["sector_id"] == "Unknown"]["series"].unique()
print("Companies without sector_id:")
for s in missing:
    print(" -", s)
