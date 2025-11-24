import pandas as pd

df = pd.read_parquet("data/tft_ready_multiseries/train.parquet")

pd.DataFrame({"series": df["series"].unique()}).to_csv("series_list.csv", index=False)
print("Saved → series_list.csv")

