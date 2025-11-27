#!/usr/bin/env python3
"""
fetch_abs_gdp_sdmx.py

Extract GDP data from ABS SDMX XML file (ANA_AGG).
Input:
    data/raw_abs/all.xml

Outputs (example):
    data/raw_abs/AUS_GDP_GPM_PCA.parquet
    data/raw_abs/AUS_GDP_GDPV.parquet
    ...

All files use schema:
    date, value, series, region, source
"""

import os
import pandas as pd
import xml.etree.ElementTree as ET

ABS_XML_PATH = "data/raw_abs/all.xml"
OUT_DIR = "data/raw_abs"


def quarter_to_date(q):
    """Convert '1974-Q3' → datetime(1974-09-30)."""
    try:
        return pd.Period(q, freq="Q").to_timestamp("Q")
    except:
        return None


def parse_abs_sdmx(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {
        "gen": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic",
    }

    series_list = root.findall(".//gen:Series", ns)

    outputs = {}  # series_name → list of (date, value)

    for s in series_list:
        # Extract SeriesKey dimensions
        keys = {}
        for kv in s.findall("gen:SeriesKey/gen:Value", ns):
            keys[kv.attrib["id"]] = kv.attrib["value"]

        data_item = keys.get("DATA_ITEM", "")
        measure = keys.get("MEASURE", "")
        tse = keys.get("TSEST", "")
        region = keys.get("REGION", "AUS")
        freq = keys.get("FREQ", "Q")

        # Create canonical series name
        canon = f"AUS_GDP_{data_item}_{measure}_{tse}_{freq}"

        # Parse observations
        for obs in s.findall("gen:Obs", ns):
            dim = obs.find("gen:ObsDimension", ns)
            val = obs.find("gen:ObsValue", ns)

            if dim is None:
                continue

            date_str = dim.attrib.get("value")
            date = quarter_to_date(date_str)

            if date is None:
                continue

            value = None
            if val is not None:
                value = val.attrib.get("value")

            outputs.setdefault(canon, []).append((date, value))

    return outputs


def main():
    print("\n[INFO] Parsing ABS GDP SDMX XML…")

    if not os.path.exists(ABS_XML_PATH):
        raise FileNotFoundError(f"Missing file: {ABS_XML_PATH}")

    series_dict = parse_abs_sdmx(ABS_XML_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    for canon, rows in series_dict.items():
        df = pd.DataFrame(rows, columns=["date", "value"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])

        df["series"] = canon
        df["region"] = "AUS"
        df["source"] = "ABS_SDMX"

        out_path = os.path.join(OUT_DIR, f"{canon}.parquet")
        df.to_parquet(out_path, index=False)

        print(f"[OK] Saved {out_path} ({len(df)} rows)")

    print("\n[COMPLETE] ABS GDP SDMX extraction finished.\n")


if __name__ == "__main__":
    main()
