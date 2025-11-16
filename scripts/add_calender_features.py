#!/usr/bin/env python3
"""
add_calendar_features.py

Adds LEVEL-1 time features (calendar & seasonality) to merged datasets.
Works for both:
    - data/merged/merged_features.parquet  (macro)
    - data/company_datasets/*.parquet      (company-level)
"""

import pandas as pd
import numpy as np
import os
from pandas.tseries.holiday import (
    USFederalHolidayCalendar,
)
import holidays  # for AU + CN holidays

MACRO_PATH = "data/merged/merged_features.parquet"
COMPANY_DIR = "data/company_datasets"
OUTPUT_DIR = "data/level1"


def ensure_date(df):
    """Guarantee a proper datetime Date column."""
    # If date is index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Find any column named "Date"
    date_cols = [c for c in df.columns if c.lower() == "date"]

    if len(date_cols) == 0:
        raise RuntimeError("No Date column found!")

    # Use the last date column if duplicates exist
    df = df.rename(columns={date_cols[-1]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def add_basic_calendar(df):
    """Add core calendar & seasonal features."""
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["day_of_month"] = df["Date"].dt.day

    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df["is_quarter_end"] = df["Date"].dt.is_quarter_end.astype(int)
    df["is_year_end"] = df["Date"].dt.is_year_end.astype(int)
    return df


def add_holiday_flags(df):
    """Add holiday flags for AU, US, CN."""
    years = list(range(df["Date"].dt.year.min(), df["Date"].dt.year.max() + 1))

    # Australia
    aus_holidays = holidays.Australia(years=years)

    # China
    china_holidays = holidays.China(years=years)

    # US (NY market + SPY relevance)
    us_holidays = USFederalHolidayCalendar().holidays(
        start=str(years[0]),
        end=str(years[-1])
    )

    df["is_aus_holiday"] = df["Date"].isin(aus_holidays).astype(int)
    df["is_china_holiday"] = df["Date"].isin(china_holidays).astype(int)
    df["is_us_holiday"] = df["Date"].isin(us_holidays).astype(int)

    return df


def add_relative_holiday_distance(df):
    """Distance to nearest holiday (past and future)."""
    df = df.sort_values("Date")
    df["is_any_holiday"] = (
        df["is_aus_holiday"] | df["is_us_holiday"] | df["is_china_holiday"]
    ).astype(int)

    # Rolling past distance
    last_holiday_date = None
    distances = []

    for d, flag in zip(df["Date"], df["is_any_holiday"]):
        if flag == 1:
            last_holiday_date = d
        if last_holiday_date is None:
            distances.append(np.nan)
        else:
            distances.append((d - last_holiday_date).days)

    df["days_since_holiday"] = distances

    # Future distance
    future_dists = []
    next_holiday_date = None

    # reverse iteration
    for d, flag in zip(df["Date"][::-1], df["is_any_holiday"][::-1]):
        if flag == 1:
            next_holiday_date = d

        if next_holiday_date is None:
            future_dists.append(np.nan)
        else:
            future_dists.append((next_holiday_date - d).days)

    df["days_until_holiday"] = future_dists[::-1]
    return df.drop(columns=["is_any_holiday"])


def enrich_file(path, output_name):
    print(f"[INFO] Enriching: {path}")

    df = pd.read_parquet(path)
    df = ensure_date(df)

    df = add_basic_calendar(df)
    df = add_holiday_flags(df)
    df = add_relative_holiday_distance(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/{output_name}"
    df.to_parquet(out_path)

    print(f"[OK] Saved enriched file: {out_path}")


def main():
    # Enrich macro dataset
    enrich_file(MACRO_PATH, "macro_enriched.parquet")

    # Enrich each company dataset
    if os.path.exists(COMPANY_DIR):
        for fname in os.listdir(COMPANY_DIR):
            if fname.endswith(".parquet"):
                enrich_file(
                    f"{COMPANY_DIR}/{fname}",
                    f"{fname.replace('.parquet','')}_enriched.parquet"
                )


if __name__ == "__main__":
    main()
# python3 scripts/add_calender_features.py
