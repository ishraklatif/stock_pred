# import pandas as pd

# df = pd.read_parquet("data/cache/AXJO_20251114_ind_20251114.parquet")

# print(df.head())
# print(df.columns)
# print(df.tail())

# import pandas as pd

# df = pd.read_parquet("data/merged/merged_features.parquet")

# print(df.head())
# print(df.tail())
# print(df.info())
# print(df.columns)

# print("✅ Option 2 — Check shape + missing values")

# print("Rows:", len(df))
# print("Columns:", len(df.columns))
# print(df.isna().sum().sort_values().tail(20))

# import pandas as pd
# df = pd.read_parquet("data/enriched/macro_level2.parquet")

# return_cols = [c for c in df.columns if "return" in c.lower()]
# for c in return_cols:
#     print(c)

# print("*************************COMPANY****************************")
# df2 = pd.read_parquet("data/enriched/ANZ.AX_dataset_enriched.parquet")

# return_cols = [c for c in df2.columns if "return" in c.lower()]
# for d in return_cols:
#     print(d)



import pandas as pd
df = pd.read_parquet("data/level4_company/CBA_AX_merged.parquet")
for col in df.columns:
    if "CBA" in col: print(col)

