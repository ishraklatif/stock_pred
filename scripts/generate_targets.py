#!/usr/bin/env python3
"""
generate_targets.py

Print all valid TARGET columns for training TFT models.
"""

import pandas as pd
import re

PATH = "data/final/macro_clean_sanitized.parquet"

def is_valid_target(col):
    """
    A target column must:
    - contain 'Close'
    - NOT contain SMA/EMA/MACD/etc indicators
    """
    if "Close" not in col:
        return False
    
    # Exclude indicator columns
    if any(x in col for x in ["SMA", "EMA", "MACD", "BB", "ATR", "vol"]):
        return False

    return True


def main():
    df = pd.read_parquet(PATH)

    targets = [c for c in df.columns if is_valid_target(c)]

    print("\n===== IDENTIFIED TARGET COLUMNS =====")
    for t in targets:
        print(t)

    print(f"\n[OK] Total targets detected: {len(targets)}")


if __name__ == "__main__":
    main()
