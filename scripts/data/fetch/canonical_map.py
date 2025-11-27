#!/usr/bin/env python3
"""
canonical_map.py

Central place for mapping multiple raw Yahoo Finance tickers
(and other aliases) to a single canonical instrument name.

Used by:
- fetch_macro_main.py
- fetch_macro_market.py
- fetch_macro_news.py

Goals:
- Handle alternate tickers (GC=F, XAUUSD, GOLDAMGBD228NLBM → GOLD)
- Ensure a single canonical ID (e.g., GOLD, DXY, AXJO, AUDUSD)
- Avoid duplicate instruments downstream (no *_x / *_y columns)
"""

from typing import Dict, List


# ---------------------------------------------------------------------
# RAW → CANONICAL TICKER MAP
# ---------------------------------------------------------------------
# This includes:
# - global equity index futures/indices
# - FX pairs
# - commodities
# - dollar index
# - some common alternates for robustness
# ---------------------------------------------------------------------
CANONICAL_MAP: Dict[str, str] = {
    # -----------------------
    # Indices (macro)
    # -----------------------
    "^AXJO": "AXJO",
    "^GSPC": "GSPC",
    "^FTSE": "FTSE",
    "^N225": "N225",
    "^HSI": "HSI",
    "^VIX":  "VIX",

    "AXJO": "AXJO",
    "GSPC": "GSPC",
    "FTSE": "FTSE",
    "N225": "N225",
    "HSI":  "HSI",
    "VIX":  "VIX",

    "000001.SS": "SSE",
    "000300.SS": "CSI300",
    "^SSE":      "SSE",
    "^CSI300":   "CSI300",


    # -----------------------
    # Dollar Index
    # -----------------------
    "DX-Y.NYB": "DXY",
    "DX_Y_NYB": "DXY",
    "DXY":      "DXY",

    # -----------------------
    # FX – AUDUSD & AUDJPY + variants
    # (you chose flexible mapping = handle alternates)
    # -----------------------
    "AUDUSD=X":      "AUDUSD",
    "AUDUSDUSD=X":   "AUDUSD",
    "AUD/USD":       "AUDUSD",
    "FX:AUDUSD":     "AUDUSD",
    "AUDUSD":        "AUDUSD",

    "AUDJPY=X":      "AUDJPY",
    "AUDJPYJPY=X":   "AUDJPY",
    "AUD/JPY":       "AUDJPY",
    "FX:AUDJPY":     "AUDJPY",
    "AUDJPY":        "AUDJPY",

    # -----------------------
    # Commodities – GOLD / COPPER / OIL / SILVER
    # with futures, spot & some common alternates
    # -----------------------
    # GOLD
    "GC=F":               "GOLD",
    "GC_F":               "GOLD",
    "XAUUSD":             "GOLD",
    "GOLDAMGBD228NLBM":   "GOLD",
    "GOLD":               "GOLD",

    # COPPER
    "HG=F":     "COPPER",
    "HG_F":     "COPPER",
    "COPPER":   "COPPER",

    # OIL (Brent/WTI, generic ticker “OIL”)
    "BZ=F":     "OIL",
    "BZ_F":     "OIL",
    "CL=F":     "OIL",
    "CL_F":     "OIL",
    "OIL":      "OIL",

    # SILVER
    "SI=F":     "SILVER",
    "SI_F":     "SILVER",
    "XAGUSD":   "SILVER",
    "SILVER":   "SILVER",

    # Broad commodities ETF / index
    "DBC":      "DBC",

    # ---------------------------------------------------------------------
    # Additional mappings for robustness
    # ---------------------------------------------------------------------

    # SPY variants
    "SPY": "SPY",
    "SPY?P=SPY": "SPY",

    # DBC variants
    "DBC=X": "DBC",
    "DBC?P=DBC": "DBC",
# NOTE:
# SSE and CSI300 mappings intentionally allow both with and without ".SS" extension.
# This is safe because canonical_name() reduces duplicates at save-time.

    # China index alternates (SSE / CSI300)
    "SSE": "SSE",
    "CSI300": "CSI300",
    "000001": "SSE",
    "000300": "CSI300",

    # Lowercase sector ETF variants
    "xlf": "XLF",
    "xlk": "XLK",
    "xli": "XLI",
    "xle": "XLE",
    "xlv": "XLV",
    "xlu": "XLU",
    "xlb": "XLB",
    "xlp": "XLP",
    "xly": "XLY",
    "xlre": "XLRE",
    "xlc": "XLC",


    # -----------------------
    # SPX alternates
    # -----------------------
    "^SPX": "GSPC",
    "SPX": "GSPC",

    # -----------------------
    # Futures alternates
    # -----------------------
    "ES=F": "GSPC",   # S&P 500 futures
    "VX=F": "VIX",    # VIX futures

    "ES": "GSPC",     # S&P500 futures (alternate code)
    "^ES": "GSPC",
    "VX": "VIX",      # VIX futures
    "^VX": "VIX",
    "DX-Y": "DXY",
    "DX-Y?P=DX-Y.NYB": "DXY",

   

    
    # =========================================================
    # ABS (Australian Bureau of Statistics) series IDs
    # =========================================================
    "A2325846C": "ABS_CPI",       # CPI All Groups
    "A84423029T": "ABS_UNEMP",    # Unemployment rate
    "A2304657V": "ABS_GDP",       # GDP chain volume

    "a2325846c": "ABS_CPI",
    "a84423029t": "ABS_UNEMP",
    "a2304657v": "ABS_GDP",

    # =========================================================
    # RBA (Reserve Bank of Australia) series IDs
    # =========================================================
    "FIRMMCRTD": "RBA_CASH_RATE",     # Cash rate target
    "Y10D": "RBA_YIELD_10Y",          # 10-year AU bond
    "Y2D": "RBA_YIELD_2Y",            # 2-year AU bond

    "firmmcrtd": "RBA_CASH_RATE",
    "y10d": "RBA_YIELD_10Y",
    "y2d": "RBA_YIELD_2Y",

    "^AU10Y": "RBA_YIELD_10Y",
    "AU10Y": "RBA_YIELD_10Y",
    "^AU2Y": "RBA_YIELD_2Y",
    "AU2Y": "RBA_YIELD_2Y",

    "2325846C": "ABS_CPI",
    "84423029T": "ABS_UNEMP",
    "2304657V": "ABS_GDP",






}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def safe_filename(name: str) -> str:
    """
    Convert a canonical name into a filesystem-safe filename.
    For canonical names, this will usually be a no-op, but is robust
    against spaces, dots, etc.
    """
    import re

    s = name.strip()
    # Replace any non-alphanumeric character with underscore
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    # Strip leading/trailing underscores
    s = s.strip("_")
    return s


def canonical_name(symbol_or_alias: str) -> str:
    """
    Given a raw Yahoo symbol or macro series ID, return a canonical instrument name.

    Rules:
      - ABS/RBA/FRED macro series (uppercase alphanumeric >= 6 chars) are returned unchanged.
      - Yahoo tickers are canonicalized using CANONICAL_MAP.
      - Otherwise fallback to safe_filename().
    """

    if symbol_or_alias is None:
        raise ValueError("symbol_or_alias cannot be None")

    s = symbol_or_alias.strip()
    u = s.upper()

    # --------------------------------------------------------
    # 1. Protect macro series (ABS, RBA, FRED)
    #    e.g., CPIAUCSL, GDPC1, PCUOMFGOMFG, A2325846C, FIRMMCRTD
    # --------------------------------------------------------
    if (
        u.isalnum()              # alphanumeric only
        and len(u) >= 6          # macro series are long
        and "." not in u         # not Yahoo tickers like 000001.SS
        and "/" not in u
        and "=" not in u
    ):
        return u   # Return as-is, uppercase normalized

    # --------------------------------------------------------
    # 2. Direct canonical map lookup
    # --------------------------------------------------------
    if s in CANONICAL_MAP:
        return CANONICAL_MAP[s]

    if u in CANONICAL_MAP:
        return CANONICAL_MAP[u]

    # --------------------------------------------------------
    # 3. Fallback: safe filename for all other tickers
    # --------------------------------------------------------
    return safe_filename(s)



def group_by_canonical(symbols: List[str]) -> Dict[str, List[str]]:
    """
    Group a list of raw Yahoo symbols by canonical instrument name.

    Example:
        ["GC=F", "XAUUSD", "GOLD"] → {"GOLD": ["GC=F", "XAUUSD", "GOLD"]}

    This is used by fetch scripts to:
    - Try multiple raw symbols for a single canonical asset
    - Then choose the best data (longest history) and save only once.
    """
    groups: Dict[str, List[str]] = {}

    for sym in symbols:
        if sym is None:
            continue
        canon = canonical_name(sym)
        groups.setdefault(canon, []).append(sym)

    return groups
