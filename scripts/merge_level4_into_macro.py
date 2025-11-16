#!/usr/bin/env python3
"""
merge_level4_into_macro.py

Merges:
  macro_full.parquet  (Level 1–3)
  + sentiment features (Level 4)

Output:
  data/final/macro_full_level4.parquet
"""

import os
import pandas as pd

MACRO_PATH = "data/level3_merged/macro_full.parquet"
NEWS_DIR   = "data/level4/sentiment"
OUTPUT_DIR = "data/level4_merged"
OUT_FILE   = f"{OUTPUT_DIR}/macro_full_level4.parquet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("[INFO] Loading macro_full...")
    macro = pd.read_parquet(MACRO_PATH)

    # FIX: macro_full already has Date as index
    if macro.index.name is None:
        print("[WARN] macro_full has no index name. Attempting to reset...")
        macro = macro.reset_index()

    # Ensure the index is datetime
    macro.index = pd.to_datetime(macro.index)

    print("[INFO] Merging Level-4 news sentiment...")
    for f in os.listdir(NEWS_DIR):
        if not f.endswith(".parquet"):
            continue

        name = f.replace("_sentiment.parquet", "")
        print(f"[INFO] Adding sentiment for {name} ...")

        path = f"{NEWS_DIR}/{f}"
        news_df = pd.read_parquet(path)

        # Ensure date is index
        news_df.index = pd.to_datetime(news_df.index)

        macro = macro.join(news_df, how="left")

    macro = macro.ffill()

    macro.to_parquet(OUT_FILE)
    print(f"[OK] Saved Level-4 merged dataset → {OUT_FILE}")
    print("[DONE] macro_full_level4 creation complete.")


if __name__ == "__main__":
    main()
# python3 scripts/merge_level4_into_macro.py