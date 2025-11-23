#!/usr/bin/env python3
"""
Hyperparameter Search for TFT (Grid Search, Clean & Consistent)
- Never drop any columns
- Replace all NaN/inf with 0
- Preserves target 'close'
"""

import os
import itertools
import yaml
import pandas as pd
import numpy as np
import torch
import warnings

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")


CONFIG_PATH = "config/config-search.yaml"
DATA_DIR = "data/tft_ready_multiseries"
TRAIN_PATH = f"{DATA_DIR}/train.parquet"
VAL_PATH = f"{DATA_DIR}/val.parquet"


class MyTFT(TemporalFusionTransformer):
    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        wd = self.hparams.optimizer_params["weight_decay"]
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)


# -------------------------
# Load Config
# -------------------------
def load_search_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg["search_space"], cfg["data"]


# -------------------------
# Clean NaN/inf → 0
# -------------------------
def clean_numeric(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# -------------------------
# Dataset Builder
# -------------------------
def build_datasets(train_df, val_df, data_cfg):

    # ALWAYS clean first
    train_df = clean_numeric(train_df)
    val_df = clean_numeric(val_df)

    # Make categorical
    for col in data_cfg.get("static_categoricals", []):
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
            val_df[col] = val_df[col].astype(str)

    target = data_cfg["target"]
    time_idx = data_cfg["time_idx"]
    group_id = data_cfg["group_id"]

    # infer numeric features
    ignore_cols = {"Date", group_id, time_idx, target, "sector_id"}
    numeric_feats = [
        c for c in train_df.columns
        if c not in ignore_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    print(f"[INFO] (search) Feature count = {len(numeric_feats)}")

    # Loss
    loss = QuantileLoss()
    output_size = len(loss.quantiles)


    training = TimeSeriesDataSet(
        train_df,
        time_idx=time_idx,
        target=target,
        group_ids=[group_id],
        max_encoder_length=data_cfg["max_encoder_length"],
        max_prediction_length=data_cfg["max_prediction_length"],

        static_categoricals=data_cfg["static_categoricals"],
        static_reals=[],

        time_varying_known_reals=data_cfg["known_future_reals"],
        time_varying_unknown_reals=[target] + numeric_feats,
        time_varying_unknown_categoricals=[],
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_df,
        stop_randomization=True,
        predict=False,
    )

    return training, validation, loss, output_size


# -------------------------
# Run One Trial
# -------------------------
def run_trial(hparams, data_cfg, trial_id):
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    # build datasets
    training, validation, loss, output_size = build_datasets(train_df, val_df, data_cfg)

    train_loader = training.to_dataloader(train=True, batch_size=hparams["batch_size"])
    val_loader   = validation.to_dataloader(train=False, batch_size=hparams["batch_size"])

    model = MyTFT.from_dataset(
        training,
        hidden_size=hparams["hidden_size"],
        hidden_continuous_size=hparams["hidden_continuous_size"],
        attention_head_size=hparams["attention_head_size"],
        dropout=hparams["dropout"],
        learning_rate=hparams["learning_rate"],
        optimizer="adamw",
        optimizer_params={"weight_decay": hparams["weight_decay"]},
        loss=loss,
        output_size=output_size,
    )

    ckpt_dir = f"checkpoints_search/run_{trial_id}"
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer = Trainer(
        max_epochs=hparams["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[
            ModelCheckpoint(dirpath=ckpt_dir, filename="tft", monitor="val_loss"),
            EarlyStopping(monitor="val_loss", patience=5)
        ],
        log_every_n_steps=50,
        fast_dev_run=1 if os.getenv("STOCKPRED_DRY_RUN") else False,
    )

    trainer.fit(model, train_loader, val_loader)
    return trainer.callback_metrics["val_loss"].item()


# -------------------------
# Main Loop
# -------------------------
def main():
    seed_everything(42)
    search_space, data_cfg = load_search_config()

    keys = list(search_space.keys())
    values = list(search_space.values())
    combos = list(itertools.product(*values))

    print(f"[INFO] Total combinations = {len(combos)}")

    if os.getenv("STOCKPRED_DRY_RUN"):
        combos = combos[:1]
        print("[INFO] DRY RUN → only 1 trial")

    results = []
    for i, combo in enumerate(combos):
        hparams = dict(zip(keys, combo))
        hparams["max_epochs"] = search_space["max_epochs"][0]

        print(f"\n[TRIAL {i}] {hparams}")

        val_loss = run_trial(hparams, data_cfg, i)

        results.append({**hparams, "val_loss": val_loss})
        pd.DataFrame(results).to_csv("hparam_results.csv", index=False)

    print("\n[INFO] DONE → results in hparam_results.csv")


if __name__ == "__main__":
    main()
