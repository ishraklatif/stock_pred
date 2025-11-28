# #!/usr/bin/env python3
# """
# inspect_abs_skipped.py

# Inspects ALL skipped ABS GDP/ToT files and prints:
# - filename
# - columns
# - head (first 5 rows)
# - unique MEASURE / INDEX / FREQ / REGION
# - TIME_PERIOD min/max
# """

# import os
# import pandas as pd
# from pathlib import Path

# SKIPPED = Path("data/raw_abs/skipped")
# OUT = Path("abs_skipped_inspect.txt")


# def load(path):
#     try:
#         return pd.read_parquet(path)
#     except Exception as e:
#         return None


# def summarize(df):
#     s = []

#     # Columns
#     s.append(f"  Columns: {list(df.columns)}")

#     # Unique code columns
#     for col in ["MEASURE", "INDEX", "FREQ", "REGION", "INDUSTRY", "PROPERTY_TYPE", "TSEST"]:
#         if col in df.columns:
#             s.append(f"  Unique {col}: {df[col].dropna().unique()[:10]}")

#     # Date range
#     if "TIME_PERIOD" in df.columns:
#         try:
#             dt = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
#             s.append(f"  Date range: {dt.min()} → {dt.max()}")
#         except:
#             s.append("  Date parsing failed")

#     # Sample
#     s.append("  Head:")
#     s.append(str(df.head()))

#     return "\n".join(s)


# def main():
#     files = sorted([f for f in SKIPPED.iterdir() if f.suffix == ".parquet"])

#     with open(OUT, "w") as fp:
#         fp.write("=== INSPECT ABS SKIPPED FILES ===\n\n")

#         for f in files:
#             fp.write(f"\n--- {f.name} ---\n")

#             df = load(f)
#             if df is None:
#                 fp.write("  [ERROR] Could not load file.\n")
#                 continue

#             fp.write(summarize(df))
#             fp.write("\n")

#     print(f"[OK] Inspection written to: {OUT}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
inspect_abs_skipped.py

Inspect ALL skipped ABS GDP/ToT-style files in:
    data/raw_abs/skipped/

Writes a detailed report to:
    logs/abs_skipped_inspect.txt

Why:
- These files are valid SDMX tables that were previously archived/skipped
  by clean_abs.py.
- This script helps you understand column names, key SDMX dimensions,
  date ranges, and candidate series identifiers.

Run:
    python3 -m scripts.data.debug.inspect_abs_skipped
"""

import os
import sys
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # project/
SKIPPED_DIR = Path("data/processed_abs/abs_all_series")
LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "abs_inspect.txt"

os.makedirs(LOG_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# UTIL: dual print (stdout + log file)
# -----------------------------------------------------------------------------
def make_printer(log_fh):
    def p(line: str = ""):
        print(line)
        log_fh.write(line + "\n")
    return p


# -----------------------------------------------------------------------------
# INSPECTION HELPERS
# -----------------------------------------------------------------------------
def summarize_series(df: pd.DataFrame, p):
    """Print quick SDMX-ish summary for an ABS table."""
    cols = df.columns.tolist()
    p(f"   Columns: {cols}")

    # Time dimension
    if "TIME_PERIOD" in df.columns:
        dates = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
        dates = dates.dropna()
        if not dates.empty:
            p(f"   TIME_PERIOD range: {dates.min().date()} → {dates.max().date()}")
        else:
            p("   TIME_PERIOD range: <all NaT>")
    else:
        p("   TIME_PERIOD: <missing>")

    # Observed values
    if "OBS_VALUE" in df.columns:
        vals = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
        p(f"   OBS_VALUE: count={vals.notna().sum()}, "
          f"min={vals.min()}, max={vals.max()}, mean={vals.mean()}")
    else:
        p("   OBS_VALUE: <missing>")

    # Key dimensions you care about for GDP/ToT
    key_dims = ["MEASURE", "INDEX", "TSEST", "REGION", "FREQ",
                "UNIT_MEASURE", "PROPERTY_TYPE", "SEX", "AGE",
                "SECTOR", "INDUSTRY"]

    for col in key_dims:
        if col in df.columns:
            uniq = df[col].dropna().unique()
            if len(uniq) > 10:
                sample = uniq[:10]
                p(f"   {col}: {len(uniq)} unique (sample={list(sample)})")
            else:
                p(f"   {col}: {list(uniq)}")


def inspect_file(path: Path, p):
    p("=" * 80)
    p(f"[FILE] {path.name}")
    p("=" * 80)
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        p(f"  [ERROR] Could not load: {e}")
        return

    if df.empty:
        p("  [WARN] EMPTY file")
        return

    p(f"   Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    summarize_series(df, p)

    # Show small head/tail slices, but keep it short
    p("\n   Head (3 rows):")
    p(df.head(3).to_string())
    p("\n   Tail (3 rows):")
    p(df.tail(3).to_string())
    p("")  # spacer


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        p = make_printer(fh)

        p("=" * 80)
        p(" INSPECTING: ABS SKIPPED GDP / ToT FILES")
        p("=" * 80)
        p(f"Root: {ROOT}")
        p(f"Skipped folder: {SKIPPED_DIR}")
        p(f"Log file: {LOG_PATH}")
        p("")

        if not SKIPPED_DIR.exists():
            p(f"[WARN] Skipped folder does NOT exist: {SKIPPED_DIR}")
            p("       Run clean_abs.py first so it archives GDP/ToT/other files.")
            return

        files = sorted([f for f in SKIPPED_DIR.glob("*.parquet")])
        if not files:
            p("[INFO] No .parquet files found in skipped folder.")
            return

        p(f"[INFO] Found {len(files)} skipped ABS files.\n")

        for fp in files:
            inspect_file(fp, p)

        p("=" * 80)
        p(" ABS SKIPPED INSPECTION COMPLETE ")
        p("=" * 80)

    print(f"\n[OK] Detailed report written to: {LOG_PATH}\n")


if __name__ == "__main__":
    # Allow module-style running via `python3 -m scripts.data.debug.inspect_abs_skipped`
    if __name__ == "__main__":
        main()
