# ============================================================================
# FILE: scripts/data/fetch/fetch_abs_labour.py
# ============================================================================

#!/usr/bin/env python3
"""Fetch ABS Labour Force data"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from abs_api_base import abs_fetch_all_data, save_parquet


def main():
    print("\n" + "="*80)
    print("[RUN] ABS Labour Force / Unemployment")
    print("="*80)
    
    print("\n[INFO] Fetching all Labour Force data...")
    df = abs_fetch_all_data("LF", start_period="2000")
    
    if df is None or df.empty:
        print("[ERROR] Failed to fetch Labour Force data")
        return
    
    print(f"\n[INFO] Labour Force Data Summary:")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Columns: {list(df.columns)}")
    
    # Save full dataset
    save_parquet(df, "data/raw_abs/AUS_LF_Full.parquet")
    
    # Try to filter for unemployment and participation rates
    try:
        keywords = ['Unemployment rate', 'Participation rate', 'Labour force']
        
        for col in df.columns:
            if df[col].dtype == 'object':
                for keyword in keywords:
                    mask = df[col].astype(str).str.contains(keyword, case=False, na=False)
                    if mask.any():
                        filtered = df[mask]
                        label = keyword.replace(' ', '_')
                        out_path = f"data/raw_abs/AUS_LF_{label}.parquet"
                        save_parquet(filtered, out_path)
                        print(f"[OK] Found {len(filtered)} rows matching '{keyword}'")
    
    except Exception as e:
        print(f"[WARN] Could not filter data: {e}")
    
    print("\n[COMPLETE] Labour Force fetch complete")


if __name__ == "__main__":
    main()