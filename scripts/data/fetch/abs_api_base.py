# ============================================================================
# FILE: scripts/data/fetch/abs_api_base.py
# ============================================================================

import os
import ssl
import pandas as pd
import requests
from io import StringIO


def abs_fetch_all_data(dataflow_id, start_period="2000", end_period=None):
    """
    Fetch ALL data from an ABS dataflow
    
    The ABS SDMX API doesn't support querying by series ID directly.
    Instead, we fetch all data and filter locally.
    
    Parameters:
    -----------
    dataflow_id : str
        Dataflow ID (e.g., 'CPI', 'LF', 'WPI')
    start_period : str
        Start period (default: '2000')
    end_period : str, optional
        End period
    
    Returns:
    --------
    pd.DataFrame or None
    """
    url = f"https://data.api.abs.gov.au/rest/data/ABS,{dataflow_id}/all"
    params = {"format": "csv"}
    
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period
    
    # Disable SSL verification if needed
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass
    
    print(f"[DEBUG] URL: {url}")
    print(f"[DEBUG] Params: {params}")
    
    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        
        df = pd.read_csv(StringIO(resp.text))
        print(f"[OK] Fetched {len(df)} rows with {len(df.columns)} columns")
        return df
        
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] HTTP {e.response.status_code}: {str(e)[:200]}")
        return None
    except Exception as e:
        print(f"[ERROR] {str(e)[:200]}")
        return None


def save_parquet(df, out_path):
    """Save DataFrame to parquet with automatic directory creation"""
    if df is None or df.empty:
        print(f"[WARN] No data to save to {out_path}")
        return
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[OK] Saved {out_path} ({len(df)} rows)")