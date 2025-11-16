#!/usr/bin/env python3
"""
score_news_sentiment.py

Loads raw news → applies FinBERT → produces daily sentiment features:
- sentiment_mean
- sentiment_max
- sentiment_min
- sentiment_volatility
- headline_count
"""

import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from datetime import datetime

RAW_DIR = "data/level4/raw_news"
OUT_DIR = "data/level4/sentiment"
os.makedirs(OUT_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("Using device:", DEVICE)


MODEL = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(DEVICE)

def score_text(text):
    """Run FinBERT sentiment analysis."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs).logits
    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    return {
        "positive": probs[0],
        "neutral": probs[1],
        "negative": probs[2],
        "sentiment_score": probs[0] - probs[2]
    }

def main():
    final_frames = {}

    for file in os.listdir(RAW_DIR):
        if not file.endswith(".json"):
            continue

        asset = file.replace("_news.json", "")
        print(f"[INFO] Processing sentiment for: {asset}")

        path = f"{RAW_DIR}/{file}"
        news = json.load(open(path))

        rows = []
        for n in news:

            content = n.get("content", {})

            title = content.get("title", "")
            summary = content.get("summary", "")

            # Combine title + summary
            text = (title + " " + summary).strip()
            if len(text) < 5:
                continue

            # Parse date safely
            pubdate = content.get("pubDate", None)
            if pubdate is None:
                continue
            try:
                dt = pd.to_datetime(pubdate)
            except:
                continue

            # Run sentiment
            sent = score_text(text)

            row = {"Date": dt, **sent}
            rows.append(row)


        if not rows:
            print(f"[WARN] No valid news for {asset}")
            continue

        df = pd.DataFrame(rows)
        df = df.groupby(df["Date"].dt.date).agg({
            "sentiment_score": ["mean", "max", "min", "std"],
            "positive": "mean",
            "neutral": "mean",
            "negative": "mean",
        })
        df.columns = [
            f"{asset}_sent_mean",
            f"{asset}_sent_max",
            f"{asset}_sent_min",
            f"{asset}_sent_vol",
            f"{asset}_pos",
            f"{asset}_neu",
            f"{asset}_neg",
        ]

        # Final daily frame
        df.index = pd.to_datetime(df.index)
        final_frames[asset] = df

        out_path = f"{OUT_DIR}/{asset}_sentiment.parquet"
        df.to_parquet(out_path)
        print(f"[OK] Saved sentiment → {out_path}")

    print("\n[DONE] All sentiment processed.\n")


if __name__ == "__main__":
    main()

# python3 scripts/score_news_sentiment.py