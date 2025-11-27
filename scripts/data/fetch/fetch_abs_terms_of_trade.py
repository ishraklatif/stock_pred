# ============================================================================
# FILE: scripts/data/fetch/fetch_abs_terms_of_trade.py
# ============================================================================

#!/usr/bin/env python3
"""Fetch ABS Terms of Trade data"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from abs_api_base import abs_fetch_all_data, save_parquet


def main():
    print("\n" + "="*80)
    print("[RUN] ABS Terms of Trade")
    print("="*80)
    
    # Try multiple possible dataflows
    possible_dataflows = [
        ("ITPI", "International Trade Price Indexes"),
        ("ITPE", "International Trade Price Exports"),
        ("IST", "International Statistics"),
        ("ITP", "International Trade Prices")
    ]
    
    success = False
    
    for dataflow_id, name in possible_dataflows:
        print(f"\n[INFO] Trying {name} ({dataflow_id})...")
        df = abs_fetch_all_data(dataflow_id, start_period="2000")
        
        if df is not None and not df.empty:
            print(f"[SUCCESS] Retrieved data from {dataflow_id}")
            print(f"  - Rows: {len(df)}")
            print(f"  - Columns: {list(df.columns)}")
            
            # Save full dataset
            save_parquet(df, f"data/raw_abs/AUS_TOT_{dataflow_id}_Full.parquet")
            
            # Try to filter for Terms of Trade
            try:
                keywords = ['Terms of trade', 'Export price', 'Import price']
                
                for col in df.columns:
                    if df[col].dtype == 'object':
                        for keyword in keywords:
                            mask = df[col].astype(str).str.contains(keyword, case=False, na=False)
                            if mask.any():
                                filtered = df[mask]
                                label = keyword.replace(' ', '_')
                                out_path = f"data/raw_abs/AUS_TOT_{label}.parquet"
                                save_parquet(filtered, out_path)
                                print(f"[OK] Found {len(filtered)} rows matching '{keyword}'")
            
            except Exception as e:
                print(f"[WARN] Could not filter data: {e}")
            
            success = True
            break  # Found data, stop trying other dataflows
        else:
            print(f"[WARN] No data from {dataflow_id}")
    
    if not success:
        print("\n[ERROR] Could not fetch Terms of Trade from any dataflow")
    else:
        print("\n[COMPLETE] Terms of Trade fetch complete")


if __name__ == "__main__":
    main()
