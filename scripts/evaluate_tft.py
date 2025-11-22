#!/usr/bin/env python3
"""
Evaluate TFT Model (Per Ticker + Per Sector)
------------------------------------------
Loads best checkpoint and computes:
- RMSE per ticker
- MAPE per ticker
- RMSE per sector
- MAPE per sector
- Overall metrics
"""

import os
import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet

DATA_DIR = "data/tft_ready_multiseries"
MODEL_DIR = "checkpoints_tft"
TEST_PATH = f"{DATA_DIR}/test.parquet"


def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat)**2))


def mape(y, yhat):
    return np.mean(np.abs((y - yhat) / y)) * 100


def main():
    print("[INFO] Loading test dataset...")
    test_df = pd.read_parquet(TEST_PATH)

    # Load best checkpoint
    best_ckpt = None
    for f in os.listdir(MODEL_DIR):
        if "best" in f and f.endswith(".ckpt"):
            best_ckpt = os.path.join(MODEL_DIR, f)
            break

    if best_ckpt is None:
        raise FileNotFoundError("No best checkpoint found in checkpoints_tft/")

    print(f"[INFO] Loading model from {best_ckpt}")
    model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)

    # Build dataset from model (ensures encoding matches)
    test_ds = TimeSeriesDataSet.from_dataset(
        model.dataset,
        test_df,
        predict=False,
        stop_randomization=True
    )

    test_loader = test_ds.to_dataloader(train=False, batch_size=64)

    preds = model.predict(test_loader).numpy()
    actuals = model.dataset.transform(test_df["close"], inverse=True).values

    test_df["pred"] = preds.flatten()
    test_df["true"] = actuals

    # Ticker-level evaluation
    ticker_metrics = []
    for ticker, group in test_df.groupby("series"):
        y = group["true"].values
        yhat = group["pred"].values
        ticker_metrics.append({
            "ticker": ticker,
            "RMSE": rmse(y, yhat),
            "MAPE": mape(y, yhat),
        })

    pd.DataFrame(ticker_metrics).to_csv("metrics_per_ticker.csv", index=False)

    # Sector-level
    sector_metrics = []
    for sector, group in test_df.groupby("sector_id"):
        y = group["true"].values
        yhat = group["pred"].values
        sector_metrics.append({
            "sector": sector,
            "RMSE": rmse(y, yhat),
            "MAPE": mape(y, yhat),
        })
    pd.DataFrame(sector_metrics).to_csv("metrics_per_sector.csv", index=False)

    print("\n[INFO] Evaluation complete.")
    print("Saved: metrics_per_ticker.csv, metrics_per_sector.csv")


if __name__ == "__main__":
    main()
