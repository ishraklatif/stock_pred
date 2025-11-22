#!/usr/bin/env python3
"""
compute_calendar_features.py

Builds a single global calendar/holiday feature table based on the union
of all dates present in:
- processed companies
- processed macro
- processed macro-market

Outputs:
    data/processed_calendar/calendar_master.parquet

Config (config/data.yaml):

data:
  processed:
    companies_folder: "data/processed_companies"
    macro_folder: "data/processed_macro"
    market_folder: "data/processed_macro_market"
    calendar_folder: "data/processed_calendar"

calendar:
  include_aus: true
  include_us: true
  include_cn: true
"""

import os
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import holidays
from pandas.tseries.holiday import USFederalHolidayCalendar

CONFIG_PATH = "config/data.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# Calendar feature helpers
# ============================================================
def add_basic_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["day_of_month"] = df["Date"].dt.day

    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df["is_quarter_end"] = df["Date"].dt.is_quarter_end.astype(int)
    df["is_year_end"] = df["Date"].dt.is_year_end.astype(int)
    return df


def add_holiday_flags(df: pd.DataFrame, cal_cfg: dict) -> pd.DataFrame:
    years = list(range(df["Date"].dt.year.min(), df["Date"].dt.year.max() + 1))

    if cal_cfg.get("include_aus", True):
        aus = holidays.Australia(years=years)
        df["is_aus_holiday"] = df["Date"].dt.date.isin(aus).astype(int)

    else:
        df["is_aus_holiday"] = 0

    if cal_cfg.get("include_cn", True):
        china = holidays.China(years=years)
        df["is_china_holiday"] = df["Date"].dt.date.isin(china).astype(int)
    else:
        df["is_china_holiday"] = 0

    if cal_cfg.get("include_us", True):
        us = USFederalHolidayCalendar().holidays(
            start=str(years[0]), end=str(years[-1])
        )
        df["is_us_holiday"] = df["Date"].dt.date.isin(us).astype(int)
    else:
        df["is_us_holiday"] = 0

    return df


def to_date_list(x):
    """
    Convert a list/Series/DatetimeIndex of dates into a list of datetime.date.
    Safe for all formats.
    """
    if x is None or len(x) == 0:
        return []

    x = pd.to_datetime(x)  # always normalize

    # DatetimeIndex → convert each element using .date()
    if isinstance(x, pd.DatetimeIndex):
        return [d.date() for d in x]

    # Series → use .dt.date
    if isinstance(x, pd.Series):
        return list(x.dt.date)

    # List of mixed values → convert individually
    return [pd.to_datetime(d).date() for d in x]


def add_holiday_distances(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add distance (in days) to nearest holiday for AUS, US, CN.
    df must contain a 'Date' column with datetime64[ns].

    Output columns:
        dist_to_aus_holiday
        dist_to_us_holiday
        dist_to_china_holiday
    """

    cal_cfg = cfg.get("calendar", {})

    # Load holiday date lists from YAML; convert safely
    aus_holidays = to_date_list(cal_cfg.get("aus_holidays", []))
    us_holidays = to_date_list(cal_cfg.get("us_holidays", []))
    china_holidays = to_date_list(cal_cfg.get("china_holidays", []))

    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Ensure df["Date"] is Python datetime.date compatible
    df["date_only"] = df["Date"].dt.date

    # Function to compute min absolute difference between one day & holiday list
    def min_dist(cur_date, holiday_list):
        if not holiday_list:
            return None
        return min(abs((cur_date - h).days) for h in holiday_list)

    df["dist_to_aus_holiday"] = df["date_only"].apply(lambda d: min_dist(d, aus_holidays))
    df["dist_to_us_holiday"] = df["date_only"].apply(lambda d: min_dist(d, us_holidays))
    df["dist_to_china_holiday"] = df["date_only"].apply(lambda d: min_dist(d, china_holidays))

    # Drop helper column
    df = df.drop(columns=["date_only"])

    return df




# ============================================================
# Main
# ============================================================
def collect_all_dates(dir_path: str) -> pd.Series:
    root = Path(dir_path)
    if not root.exists():
        return pd.Series(dtype="datetime64[ns]")

    dates = []
    for f in root.glob("*.parquet"):
        df = pd.read_parquet(f, columns=["Date"])
        if "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        dates.append(df["Date"].dropna())

    if not dates:
        return pd.Series(dtype="datetime64[ns]")

    all_dates = pd.concat(dates).dropna().unique()
    return pd.Series(all_dates)


def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    processed = data_cfg["processed"]
    cal_cfg = cfg.get("calendar", {})

    comp_dir = processed["companies_folder"]
    macro_dir = processed["macro_folder"]
    market_dir = processed["market_folder"]
    cal_out_dir = processed["calendar_folder"]

    os.makedirs(cal_out_dir, exist_ok=True)

    print("[INFO] Collecting all dates from processed datasets...")
    dates_comp = collect_all_dates(comp_dir)
    dates_macro = collect_all_dates(macro_dir)
    dates_market = collect_all_dates(market_dir)

    all_dates = pd.concat([dates_comp, dates_macro, dates_market]).dropna().unique()
    all_dates = pd.to_datetime(all_dates)
    all_dates = np.sort(all_dates)

    if len(all_dates) == 0:
        print("[WARN] No dates found in processed datasets.")
        return

    cal_df = pd.DataFrame({"Date": all_dates})

    print("[INFO] Adding calendar & holiday features...")
    cal_df = add_basic_calendar(cal_df)
    cal_df = add_holiday_flags(cal_df, cal_cfg)
    cal_df = add_holiday_distances(cal_df, cfg)

    out_path = os.path.join(cal_out_dir, "calendar_master.parquet")
    cal_df.to_parquet(out_path, index=False)
    print(f"[OK] Saved calendar_master → {out_path}")
    print("[DONE] Calendar feature generation complete.")


if __name__ == "__main__":
    main()
