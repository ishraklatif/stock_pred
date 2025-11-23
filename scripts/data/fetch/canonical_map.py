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
    Given a raw Yahoo symbol or an alias, return a canonical instrument name.

    Logic:
    1) Exact match in CANONICAL_MAP
    2) Uppercase match in CANONICAL_MAP
    3) Fallback: sanitized symbol used as canonical
    """
    if symbol_or_alias is None:
        raise ValueError("symbol_or_alias cannot be None")

    s = symbol_or_alias.strip()

    # Direct lookup
    if s in CANONICAL_MAP:
        return CANONICAL_MAP[s]

    # Uppercase lookup as backup (handles e.g. 'gold' → 'GOLD' if present)
    u = s.upper()
    if u in CANONICAL_MAP:
        return CANONICAL_MAP[u]

    # Fallback: just tidy the original
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
