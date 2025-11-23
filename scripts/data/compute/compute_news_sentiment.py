#!/usr/bin/env python3
"""
compute_news_sentiment.py

Processes raw JSON news → FinBERT → daily sentiment features.

Reads from config/data.yaml:

data:
  sources:
    news_folder: "data/news/raw"

  processed:
    news_sentiment_folder: "data/news/sentiment"

News files are expected to be named by *canonical* asset id, e.g.:

    AXJO.json, GSPC.json, CSI300.json, GOLD.json, ...

This script:

  - Loads each JSON file
  - Runs FinBERT sentiment on (title + summary)
  - Aggregates per calendar Date (normalized to midnight)
  - Writes daily sentiment parquet with columns:

      <ASSET>_sent_mean
      <ASSET>_sent_max
      <ASSET>_sent_min
      <ASSET>_sent_vol    (std)
      <ASSET>_pos
      <ASSET>_neu
      <ASSET>_neg

No forward-filling is done here.
Missing days are left as NaN and will be handled by safe_merge_sentiment
during the merge stage.
"""

import os
import json

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from scripts.data.fetch.canonical_map import CANONICAL_MAP as canonical_map  # canonical-aware

CONFIG_PATH = "config/data.yaml"
MODEL_NAME = "ProsusAI/finbert"


# ======================================================================
# CONFIG HELPERS
# ======================================================================

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ======================================================================
# CANONICAL MAPPING HELPERS
# ======================================================================

def build_reverse_canonical_map():
    """
    Build a reverse lookup: raw/alt symbol → canonical id.

    canonical_map is expected to be something like:

        {
            "GSPC": ["^GSPC", "SPY"],
            "SSE": ["000001.SS"],
            "CSI300": ["000300.SS"],
            "DXY": ["DX-Y.NYB"],
            ...
        }

    or values may be dicts with 'symbols' / 'aliases'.

    We try to be robust to both patterns.
    """
    rev = {}

    for canon, v in canonical_map.items():
        # Case 1: list/tuple/set of aliases
        if isinstance(v, (list, tuple, set)):
            for sym in v:
                rev[str(sym)] = canon

        # Case 2: single string
        elif isinstance(v, str):
            rev[v] = canon

        # Case 3: dict with possible alias lists
        elif isinstance(v, dict):
            for key in ("symbols", "aliases"):
                if key in v and isinstance(v[key], (list, tuple, set)):
                    for sym in v[key]:
                        rev[str(sym)] = canon
        # Otherwise, ignore

        # Canonical id maps to itself
        rev[str(canon)] = canon

    return rev


REVERSE_CANONICAL = build_reverse_canonical_map()


def to_canonical(asset_name: str) -> str:
    """
    Given an asset identifier (e.g. filename stem or raw symbol),
    map to canonical id if possible, otherwise return as-is.
    """
    asset_name = str(asset_name)
    if asset_name in REVERSE_CANONICAL:
        return REVERSE_CANONICAL[asset_name]
    return asset_name


# ======================================================================
# DEVICE / MODEL
# ======================================================================

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()
print(f"[INFO] Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


# ======================================================================
# SENTIMENT SCORING
# ======================================================================

def score_text(text: str):
    """Run FinBERT sentiment on a single text snippet."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    # FinBERT label order: [positive, neutral, negative]
    return {
        "positive": float(probs[0]),
        "neutral": float(probs[1]),
        "negative": float(probs[2]),
        "sentiment_score": float(probs[0] - probs[2]),
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    raw_dir = data_cfg["sources"]["news_folder"]
    out_dir = data_cfg["processed"]["news_sentiment_folder"]

    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    print(f"[INFO] Found {len(files)} raw news files in {raw_dir}")

    for file in files:
        # The file name is expected to be the canonical asset id, e.g. "AXJO.json"
        asset_raw = file.replace(".json", "")
        asset = to_canonical(asset_raw)

        path = os.path.join(raw_dir, file)
        print(f"[INFO] Processing sentiment for: {asset} (from file {file})")

        try:
            with open(path, "r") as f:
                news = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load {file}: {e}")
            continue

        rows = []
        for n in news:
            content = n.get("content", {})
            title = content.get("title", "") or ""
            summary = content.get("summary", "") or ""
            text = (title + " " + summary).strip()

            if len(text) < 5:
                continue

            pub = content.get("pubDate", None)
            if pub is None:
                continue

            dt = pd.to_datetime(pub)

            if getattr(dt, "tzinfo", None) is not None:
                try:
                    dt = dt.tz_localize(None)
                except:
                    dt = dt.tz_convert(None).tz_localize(None)



            sent = score_text(text)
            rows.append({
                "Date": dt,
                "sentiment_score": sent["sentiment_score"],
                "positive": sent["positive"],
                "neutral": sent["neutral"],
                "negative": sent["negative"],
            })

        if not rows:
            print(f"[WARN] No valid news items for {asset}")
            continue

        df = pd.DataFrame(rows)

        # Normalize Date to midnight so grouping is stable
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

        # Daily aggregation
        daily = df.groupby("Date", as_index=False).agg(
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_max=("sentiment_score", "max"),
            sentiment_min=("sentiment_score", "min"),
            sentiment_vol=("sentiment_score", "std"),
            positive_mean=("positive", "mean"),
            neutral_mean=("neutral", "mean"),
            negative_mean=("negative", "mean"),
        )

        # Rename columns to canonical Option A:
        #   <ASSET>_sent_mean, <ASSET>_sent_max, ...
        daily = daily.rename(columns={
            "sentiment_mean": f"{asset}_sent_mean",
            "sentiment_max": f"{asset}_sent_max",
            "sentiment_min": f"{asset}_sent_min",
            "sentiment_vol": f"{asset}_sent_vol",
            "positive_mean": f"{asset}_pos",
            "neutral_mean": f"{asset}_neu",
            "negative_mean": f"{asset}_neg",
        })

        # Ensure Date is the first column
        cols = ["Date"] + [c for c in daily.columns if c != "Date"]
        daily = daily[cols]

        out_path = os.path.join(out_dir, f"{asset}.parquet")
        daily.to_parquet(out_path, index=False)
        print(f"[OK] Saved → {out_path}")

    print("[DONE] All news sentiment computed.")


if __name__ == "__main__":
    main()
