import pandas as pd

df = pd.read_parquet("data/tft_ready_multiseries/train.parquet")

pd.DataFrame({"column_name": df.columns}).to_csv(
    "train_columns.csv",
    index=False
)

print("Saved → train_columns.csv")
