#!/usr/bin/env python3
"""
compute_calendar_features.py

Generates global calendar/holiday features on a continuous date range.
No dropping of dates. Compatible with TFT multiseries pipeline.
"""

import os
import yaml
import numpy as np
import pandas as pd
import holidays
from pandas.tseries.holiday import USFederalHolidayCalendar

CONFIG_PATH = "config/data.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
# BASIC CALENDAR FEATURES
# ----------------------------------------------------------------------
def add_basic_calendar(df):
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["day_of_month"] = df["Date"].dt.day

    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df["is_quarter_end"] = df["Date"].dt.is_quarter_end.astype(int)
    df["is_year_end"] = df["Date"].dt.is_year_end.astype(int)
    return df


# ----------------------------------------------------------------------
# HOLIDAY FLAGS
# ----------------------------------------------------------------------
def add_holiday_flags(df, cfg):
    years = range(df["Date"].dt.year.min(), df["Date"].dt.year.max() + 1)

    if cfg.get("include_aus", True):
        aus = holidays.Australia(years=years)
        df["is_aus_holiday"] = df["Date"].dt.date.isin(aus).astype(int)
    else:
        df["is_aus_holiday"] = 0

    if cfg.get("include_cn", True):
        cn = holidays.China(years=years)
        df["is_china_holiday"] = df["Date"].dt.date.isin(cn).astype(int)
    else:
        df["is_china_holiday"] = 0

    if cfg.get("include_us", True):
        us = USFederalHolidayCalendar().holidays(
            start=str(min(years)), end=str(max(years))
        )
        df["is_us_holiday"] = df["Date"].dt.date.isin(us).astype(int)
    else:
        df["is_us_holiday"] = 0

    return df


# ----------------------------------------------------------------------
# DISTANCE TO HOLIDAYS
# ----------------------------------------------------------------------
def add_holiday_distances(df, cfg):
    df = df.copy()
    df = df.sort_values("Date")
    df["date_only"] = df["Date"].dt.date

    def make_dist(col):
        mask = df[col] == 1
        if not mask.any():
            return pd.Series(np.nan, index=df.index)

        hol = df.loc[mask, "date_only"].unique()
        ords = np.array([d.toordinal() for d in hol])

        return df["date_only"].apply(
            lambda d: int(np.min(np.abs(ords - d.toordinal())))
        )

    if cfg.get("include_aus", True):
        df["dist_to_aus_holiday"] = make_dist("is_aus_holiday")
    if cfg.get("include_us", True):
        df["dist_to_us_holiday"] = make_dist("is_us_holiday")
    if cfg.get("include_cn", True):
        df["dist_to_china_holiday"] = make_dist("is_china_holiday")

    return df.drop(columns=["date_only"])


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    cfg = load_config()
    cal_cfg = cfg.get("calendar", {})
    processed_dir = cfg["data"]["processed"]["calendar_folder"]
    os.makedirs(processed_dir, exist_ok=True)

    # Build continuous B-day calendar
    start = pd.to_datetime(cfg["data"]["start_date"])
    end = pd.to_datetime(cfg["data"]["end_date"] or pd.Timestamp.today())

    print(f"[INFO] Generating calendar: {start.date()} → {end.date()}")
    dates = pd.date_range(start, end, freq="B")
    df = pd.DataFrame({"Date": dates})

    df = add_basic_calendar(df)
    df = add_holiday_flags(df, cal_cfg)
    df = add_holiday_distances(df, cal_cfg)

    out = os.path.join(processed_dir, "calendar_master.parquet")
    df.to_parquet(out, index=False)
    print(f"[OK] Saved calendar_master → {out}")


if __name__ == "__main__":
    main()
