#!/usr/bin/env python3
"""
Evaluate TFT on Reduced Feature Dataset
"""

import os
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

DATA_DIR = "data/tft_reduced"
MODEL_DIR = "checkpoints_tft"
TEST_PATH = f"{DATA_DIR}/test.parquet"


def rmse(y, yhat): return np.sqrt(np.mean((y - yhat) ** 2))
def mape(y, yhat): return np.mean(np.abs((y - yhat) / y)) * 100


def main():
    print("[INFO] Loading reduced test dataset...")
    test_df = pd.read_parquet(TEST_PATH)

    ckpt = None
    for f in os.listdir(MODEL_DIR):
        if "best" in f or "last" in f:
            ckpt = os.path.join(MODEL_DIR, f)
            break
    if ckpt is None:
        raise RuntimeError("No checkpoint found")

    print(f"[INFO] Loading model from: {ckpt}")
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt)

    test_ds = TimeSeriesDataSet.from_dataset(
        model.dataset, test_df, predict=False, stop_randomization=True
    )
    loader = test_ds.to_dataloader(train=False, batch_size=64)

    preds = model.predict(loader).numpy().flatten()
    actual = model.dataset.transform(test_df["close"], inverse=True).values

    test_df["pred"] = preds
    test_df["true"] = actual

    ticker_metrics = []
    for ticker, g in test_df.groupby("series"):
        ticker_metrics.append({
            "ticker": ticker,
            "RMSE": rmse(g["true"], g["pred"]),
            "MAPE": mape(g["true"], g["pred"]),
        })
    pd.DataFrame(ticker_metrics).to_csv("metrics_per_ticker.csv", index=False)

    sector_metrics = []
    for sector, g in test_df.groupby("sector_id"):
        sector_metrics.append({
            "sector": sector,
            "RMSE": rmse(g["true"], g["pred"]),
            "MAPE": mape(g["true"], g["pred"]),
        })
    pd.DataFrame(sector_metrics).to_csv("metrics_per_sector.csv", index=False)

    print("[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()
