# ============================================================================
# FILE: scripts/data/fetch/fetch_abs_hpi.py
# ============================================================================

#!/usr/bin/env python3
"""Fetch ABS House Price Index data"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from abs_api_base import abs_fetch_all_data, save_parquet


def main():
    print("\n" + "="*80)
    print("[RUN] ABS House Price Index (HPI)")
    print("="*80)
    
    # Try multiple possible dataflow names
    possible_dataflows = [
        ("RES_PROP", "Residential Property Prices"),
        ("HPI", "House Price Index"),
        ("RPPI", "Residential Property Price Index"),
        ("RES_DWELL", "Residential Dwellings")
    ]
    
    for dataflow_id, name in possible_dataflows:
        print(f"\n[INFO] Trying {name} ({dataflow_id})...")
        df = abs_fetch_all_data(dataflow_id, start_period="2000")
        
        if df is not None and not df.empty:
            save_parquet(df, f"data/raw_abs/AUS_HPI_{dataflow_id}_Full.parquet")
            print(f"[OK] Fetched {len(df)} records from {dataflow_id}")
            print("[COMPLETE] HPI fetch complete")
            return
        else:
            print(f"[WARN] No data from {dataflow_id}")
    
    print("\n[WARN] Could not fetch HPI from any known dataflow")


if __name__ == "__main__":
    main()