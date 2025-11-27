# ============================================================================
# FILE: scripts/data/fetch/fetch_abs_wpi.py
# ============================================================================

#!/usr/bin/env python3
"""Fetch ABS Wage Price Index data"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from abs_api_base import abs_fetch_all_data, save_parquet


def main():
    print("\n" + "="*80)
    print("[RUN] ABS Wage Price Index (WPI)")
    print("="*80)
    
    print("\n[INFO] Fetching WPI data...")
    df = abs_fetch_all_data("WPI", start_period="2000")
    
    if df is not None and not df.empty:
        save_parquet(df, "data/raw_abs/AUS_WPI_Full.parquet")
        print(f"[OK] Fetched {len(df)} WPI records")
        print("[COMPLETE] WPI fetch complete")
    else:
        print("[ERROR] Failed to fetch WPI data")


if __name__ == "__main__":
    main()