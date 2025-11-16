#!/usr/bin/env python3
"""
test_tft.py

Loads the trained TFT + test dataset.
Produces predictions + test metrics.

Outputs:
    models/tft/tft_test_predictions.csv
    models/tft/tft_test_metrics.json
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer
import pytorch_lightning as pl

DATA_DIR = "data/tft"
MODEL_DIR = "models/tft"

# Device setup
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print("[INFO] Using:", DEVICE)

def main():

    print("[INFO] Loading test dataset...")
    test = pickle.load(open(f"{DATA_DIR}/test.pkl", "rb"))

    test_loader = test.to_dataloader(train=False, batch_size=64, num_workers=0)

    print("[INFO] Loading trained TFT model...")
    model_path = f"{MODEL_DIR}/tft_model.ckpt"
    tft = TemporalFusionTransformer.load_from_checkpoint(model_path)

    print("[INFO] Predicting on test set...")
    preds, x = tft.predict(test_loader, return_x=True)

    # Extract true values
    actuals = x["decoder_target"]
    actual = np.array([a[-1] for a in actuals])  # final step in decoder

    pred = preds.flatten()

    df = pd.DataFrame({"actual": actual, "pred": pred})
    out_csv = f"{MODEL_DIR}/tft_test_predictions.csv"
    df.to_csv(out_csv, index=False)
    print("[OK] Test predictions saved:", out_csv)

    mse = float(np.mean((pred - actual)**2))
    mae = float(np.mean(np.abs(pred - actual)))

    metrics = {"test_MSE": mse, "test_MAE": mae}
    json.dump(metrics, open(f"{MODEL_DIR}/tft_test_metrics.json", "w"), indent=2)

    print("[OK] Test metrics saved:", metrics)

if __name__ == "__main__":
    main()
