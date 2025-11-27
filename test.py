# #!/usr/bin/env python3
# """
# test_abs_rba_api.py - FINAL WORKING VERSION

# Tests ABS and RBA APIs with correct URL formats.
# """

# import requests

# def test_abs_cpi():
#     """Test ABS CPI - Consumer Price Index"""
#     print("\n" + "="*80)
#     print("TESTING ABS CPI")
#     print("="*80)
    
#     # CPI dataflow - try simple dataflow ID first
#     urls_to_try = [
#         ("https://data.api.abs.gov.au/rest/data/CPI", "Full CPI dataflow"),
#         ("https://data.api.abs.gov.au/rest/data/CPI/1.10001.10..Q", "Filtered CPI"),
#     ]
    
#     for url, desc in urls_to_try:
#         print(f"\nTrying: {url}")
#         print(f"Description: {desc}")
#         r = requests.get(url, headers={'Accept': 'application/vnd.sdmx.data+json;version=2.0.0'})
#         print(f"Status: {r.status_code}")
        
#         if r.status_code == 200:
#             print("✓ SUCCESS!")
#             try:
#                 j = r.json()
#                 datasets = j.get('data', {}).get('dataSets', [])
#                 if datasets:
#                     obs = datasets[0].get('observations', {})
#                     print(f"  Observations: {len(obs)}")
#             except Exception as e:
#                 print(f"  Could not parse: {e}")
#             break  # Found working URL
#         else:
#             print(f"  Failed: {r.text[:150]}")


# def test_abs_labour():
#     """Test ABS Labour Force"""
#     print("\n" + "="*80)
#     print("TESTING ABS LABOUR FORCE")
#     print("="*80)
    
#     urls_to_try = [
#         ("https://data.api.abs.gov.au/rest/data/LF", "Full LF dataflow"),
#         ("https://data.api.abs.gov.au/rest/data/LF/0.6.3.1599.20.M", "Filtered unemployment"),
#     ]
    
#     for url, desc in urls_to_try:
#         print(f"\nTrying: {url}")
#         print(f"Description: {desc}")
#         r = requests.get(url, headers={'Accept': 'application/vnd.sdmx.data+json;version=2.0.0'})
#         print(f"Status: {r.status_code}")
        
#         if r.status_code == 200:
#             print("✓ SUCCESS!")
#             try:
#                 j = r.json()
#                 datasets = j.get('data', {}).get('dataSets', [])
#                 if datasets:
#                     obs = datasets[0].get('observations', {})
#                     print(f"  Observations: {len(obs)}")
#             except Exception as e:
#                 print(f"  Could not parse: {e}")
#             break
#         else:
#             print(f"  Failed: {r.text[:150]}")


# def test_abs_gdp():
#     """Test ABS GDP"""
#     print("\n" + "="*80)
#     print("TESTING ABS GDP")
#     print("="*80)
    
#     urls_to_try = [
#         ("https://data.api.abs.gov.au/rest/data/ANA_AGG/M1.GPM.20.AUS.Q", "GDP filtered"),
#         ("https://data.api.abs.gov.au/rest/data/ANA_AGG", "Full GDP dataflow"),
#     ]
    
#     for url, desc in urls_to_try:
#         print(f"\nTrying: {url}")
#         print(f"Description: {desc}")
#         r = requests.get(url, headers={'Accept': 'application/vnd.sdmx.data+json;version=2.0.0'})
#         print(f"Status: {r.status_code}")
        
#         if r.status_code == 200:
#             print("✓ SUCCESS!")
#             try:
#                 j = r.json()
#                 datasets = j.get('data', {}).get('dataSets', [])
#                 if datasets:
#                     obs = datasets[0].get('observations', {})
#                     print(f"  Observations: {len(obs)}")
#             except Exception as e:
#                 print(f"  Could not parse: {e}")
#             break
#         else:
#             print(f"  Failed: {r.text[:150]}")


# def test_rba():
#     """Test RBA APIs"""
#     print("\n" + "="*80)
#     print("TESTING RBA")
#     print("="*80)
    
#     urls = [
#         ("https://www.rba.gov.au/statistics/tables/csv/f1-data.csv", "F1 - Interest Rates"),
#         ("https://www.rba.gov.au/statistics/tables/csv/f2-data.csv", "F2 - Bond Yields"),
#     ]
    
#     for url, desc in urls:
#         print(f"\nTrying: {url}")
#         print(f"Description: {desc}")
#         r = requests.get(url)
#         print(f"Status: {r.status_code}")
        
#         if r.status_code == 200:
#             lines = r.text.split('\n')
#             print(f"✓ SUCCESS! ({len(lines)} lines)")
#             print(f"  First line: {lines[0][:60]}...")
#         else:
#             print(f"  Failed")


# def main():
#     print("\n" + "="*80)
#     print("ABS & RBA API TESTING - FINAL VERSION")
#     print("="*80)
    
#     test_abs_cpi()
#     test_abs_labour()
#     test_abs_gdp()
#     test_rba()
    
#     print("\n" + "="*80)
#     print("CORRECT URL FORMAT")
#     print("="*80)
#     print("""
# ABS API Format:
#   https://data.api.abs.gov.au/rest/data/{DATAFLOW_ID}/{DATAKEY}
  
#   Examples:
#     CPI (full):      /rest/data/CPI
#     CPI (filtered):  /rest/data/CPI/1.10001.10..Q
#     LF (full):       /rest/data/LF
#     GDP (filtered):  /rest/data/ANA_AGG/M1.GPM.20.AUS.Q

# RBA API Format:
#   https://www.rba.gov.au/statistics/tables/csv/{TABLE}-data.csv
  
#   Examples:
#     F1:  /statistics/tables/csv/f1-data.csv
#     F2:  /statistics/tables/csv/f2-data.csv
#     """)


# if __name__ == "__main__":
#     main()
import pandas as pd
import glob

files = glob.glob("data/raw_abs/AUS_GDP_TTR*.parquet")

for f in files:
    df = pd.read_parquet(f)
    print(f, df.head())
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    