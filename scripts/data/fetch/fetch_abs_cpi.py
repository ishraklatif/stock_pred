# ============================================================================
# FILE: scripts/data/fetch/fetch_abs_cpi.py
# ============================================================================

#!/usr/bin/env python3
"""Fetch ABS CPI data"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from abs_api_base import abs_fetch_all_data, save_parquet


def main():
    print("\n" + "="*80)
    print("[RUN] ABS CPI (Inflation)")
    print("="*80)
    
    # Fetch ALL CPI data
    print("\n[INFO] Fetching all CPI data...")
    df = abs_fetch_all_data("CPI", start_period="2000")
    
    if df is None or df.empty:
        print("[ERROR] Failed to fetch CPI data")
        return
    
    print(f"\n[INFO] CPI Data Summary:")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Columns: {list(df.columns)}")
    
    # Save full dataset
    save_parquet(df, "data/raw_abs/AUS_CPI_Full.parquet")
    
    # Analyze structure to help with filtering
    print(f"\n[INFO] Data Structure Analysis:")
    
    # Check for measure column
    measure_cols = [col for col in df.columns if 'MEASURE' in col.upper()]
    if measure_cols:
        for col in measure_cols:
            unique_vals = df[col].unique()
            print(f"  - {col}: {len(unique_vals)} unique values")
            print(f"    Sample: {list(unique_vals[:10])}")
    
    # Try to filter for key CPI measures
    try:
        # Look for All Groups CPI, Trimmed Mean, Weighted Median
        keywords = ['All groups', 'Trimmed mean', 'Weighted median', 'CPI']
        
        for col in df.columns:
            if df[col].dtype == 'object':  # String column
                for keyword in keywords:
                    mask = df[col].astype(str).str.contains(keyword, case=False, na=False)
                    if mask.any():
                        filtered = df[mask]
                        label = keyword.replace(' ', '_')
                        out_path = f"data/raw_abs/AUS_CPI_{label}.parquet"
                        save_parquet(filtered, out_path)
                        print(f"[OK] Found {len(filtered)} rows matching '{keyword}'")
    
    except Exception as e:
        print(f"[WARN] Could not filter data: {e}")
    
    print("\n[COMPLETE] CPI fetch complete")
    print("[NOTE] Review AUS_CPI_Full.parquet to identify correct series")


if __name__ == "__main__":
    main()
