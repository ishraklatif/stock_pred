import pandas as pd
from pathlib import Path

file = Path("data/cache/AXJO_20251114_ind_20251114.parquet")

df = pd.read_parquet(file)

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== HEAD (5) ===")
print(df.head())

print("\n=== TAIL (5) ===")
print(df.tail())

print("\n=== INDEX INFO ===")
print("Index Name:", df.index.name)
print("Index dtype:", df.index.dtype)
print("Index sample:", df.index[:5])

print("\n=== ANY DATETIME COLUMNS? ===")
date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
print(date_cols)

print("\n=== DTYPES ===")
print(df.dtypes)
