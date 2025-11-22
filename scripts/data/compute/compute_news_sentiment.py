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
"""

import os
import json

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CONFIG_PATH = "config/data.yaml"
MODEL_NAME = "ProsusAI/finbert"


def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def score_text(text: str):
    """Run FinBERT sentiment."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return {
        "positive": float(probs[0]),
        "neutral": float(probs[1]),
        "negative": float(probs[2]),
        "sentiment_score": float(probs[0] - probs[2]),
    }


def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    raw_dir = data_cfg["sources"]["news_folder"]
    out_dir = data_cfg["processed"]["news_sentiment_folder"]

    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    print(f"[INFO] Found {len(files)} raw news files in {raw_dir}")

    for file in files:
        asset = file.replace(".json", "")
        path = os.path.join(raw_dir, file)

        print(f"[INFO] Processing sentiment for: {asset}")

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

            try:
                dt = pd.to_datetime(pub)
            except Exception:
                continue

            sent = score_text(text)
            rows.append({"Date": dt, **sent})

        if not rows:
            print(f"[WARN] No valid news items for {asset}")
            continue

        df = pd.DataFrame(rows)

        daily = df.groupby(df["Date"].dt.date).agg(
            {
                "sentiment_score": ["mean", "max", "min", "std"],
                "positive": "mean",
                "neutral": "mean",
                "negative": "mean",
            }
        )

        daily.columns = [
            f"{asset}_sent_mean",
            f"{asset}_sent_max",
            f"{asset}_sent_min",
            f"{asset}_sent_vol",
            f"{asset}_pos",
            f"{asset}_neu",
            f"{asset}_neg",
        ]

        daily.index = pd.to_datetime(daily.index)

        out_path = os.path.join(out_dir, f"{asset}.parquet")

        daily.to_parquet(out_path)
        print(f"[OK] Saved → {out_path}")

    print("[DONE] All news sentiment computed.")


if __name__ == "__main__":
    main()
