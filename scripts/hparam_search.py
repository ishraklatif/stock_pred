#!/usr/bin/env python3
"""
Hyperparameter Search for Temporal Fusion Transformer (Grid Search)

- Loads grid from config/config-search.yaml
- Evaluates each combination on validation loss
- Supports DRY RUN mode via:
      STOCKPRED_DRY_RUN=1 python -m scripts.hparam_search

- Uses the same MyTFT optimizer override as train.py
"""

import os
import itertools
import yaml
import pandas as pd
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


# ----------------------------------------------------------------------
# MyTFT — SAME as train.py (ensures optimizer consistency)
# ----------------------------------------------------------------------
class MyTFT(TemporalFusionTransformer):
    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        wd = self.hparams.optimizer_params["weight_decay"]

        # DO NOT double-inject weight_decay
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)


# ----------------------------------------------------------------------
# Load Search Config
# ----------------------------------------------------------------------
def load_search_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    if "search_space" not in cfg or "data" not in cfg:
        raise ValueError("config-search.yaml missing required keys")

    if not isinstance(cfg["search_space"], dict):
        raise ValueError("search_space must be a dictionary")

    return cfg["search_space"], cfg["data"]


# ----------------------------------------------------------------------
# Build Dataset for One Trial
# ----------------------------------------------------------------------
def build_datasets(train_df, val_df, data_cfg):
    target = data_cfg["target"]
    time_idx = data_cfg["time_idx"]
    group_id = data_cfg["group_id"]

    # Infer numeric features
    ignore = ["Date", group_id, time_idx, target, "sector_id"]
    numeric_feats = [
        c for c in train_df.columns
        if pd.api.types.is_numeric_dtype(train_df[c]) and c not in ignore
    ]

    print(f"[INFO] (search) Initial numeric feature count = {len(numeric_feats)}")

    # Drop NaN columns (ensures zero infinite/NaN issues)
    for col in numeric_feats:
        if train_df[col].isna().any():
            train_df[col] = train_df[col].fillna(0)
        if val_df[col].isna().any():
            val_df[col] = val_df[col].fillna(0)

    print(f"[INFO] (search) Final observed historical features = {len(numeric_feats)}")

    # Loss
    loss = QuantileLoss()
    out_size = len(loss.quantiles)

    # Base dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx=time_idx,
        target=target,
        group_ids=[group_id],

        max_encoder_length=data_cfg["max_encoder_length"],
        max_prediction_length=data_cfg["max_prediction_length"],

        static_categoricals=data_cfg["static_categoricals"],

        time_varying_known_reals=data_cfg["known_future_reals"],
        time_varying_unknown_reals=[target] + numeric_feats,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_df,
        stop_randomization=True,
        predict=False,
    )

    return training, validation, loss, out_size


# ----------------------------------------------------------------------
# Run One Trial
# ----------------------------------------------------------------------
def run_trial(hparams, data_cfg, trial_id):
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df   = pd.read_parquet(VAL_PATH)

    # Fix categorical types
    train_df["series"] = train_df["series"].astype(str)
    train_df["sector_id"] = train_df["sector_id"].astype(str)
    val_df["series"] = val_df["series"].astype(str)
    val_df["sector_id"] = val_df["sector_id"].astype(str)

    # Build datasets
    training, validation, loss, output_size = build_datasets(train_df, val_df, data_cfg)

    train_loader = training.to_dataloader(
        train=True,
        batch_size=hparams["batch_size"],
    )

    val_loader = validation.to_dataloader(
        train=False,
        batch_size=hparams["batch_size"],
    )

    # Build model
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

    # Output directory
    ckpt_dir = f"checkpoints_search/run_{trial_id}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="tft",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )

    # Device
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        max_epochs=hparams["max_epochs"],
        accelerator=accelerator,
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        gradient_clip_val=0.1,
        log_every_n_steps=20,
        fast_dev_run=1 if os.getenv("STOCKPRED_DRY_RUN") else False,
    )

    trainer.fit(model, train_loader, val_loader)

    # Extract best val loss
    best_loss = trainer.callback_metrics["val_loss"].item()
    return best_loss


# ----------------------------------------------------------------------
# Main Search Loop
# ----------------------------------------------------------------------
def main():
    seed_everything(42)
    search_space, data_cfg = load_search_config()

    keys = list(search_space.keys())
    values = list(search_space.values())

    combos = list(itertools.product(*values))
    print(f"[INFO] Total combinations from search_space: {len(combos)}")

    # DRY RUN limit
    if os.getenv("STOCKPRED_DRY_RUN"):
        print("[INFO] DRY RUN mode → limiting to first combination only.")
        combos = combos[:1]

    results = []

    for i, combo in enumerate(combos):
        hparams = dict(zip(keys, combo))
        print(f"[TRIAL {i}] Hyperparameters: {hparams}")

        # Extract max_epochs
        hparams["max_epochs"] = search_space["max_epochs"][0]

        val_loss = run_trial(hparams, data_cfg, i)

        print(f"[RESULT] Trial {i} → val_loss = {val_loss}")
        results.append({**hparams, "val_loss": val_loss})

        pd.DataFrame(results).to_csv("hparam_results.csv", index=False)

    print("\n[INFO] Search complete → results saved to hparam_results.csv")


if __name__ == "__main__":
    main()



# # Dry-run training (single fast_dev_run)
# STOCKPRED_DRY_RUN=1 python -m scripts.train

# # Dry-run hyperparameter search (first combination only, fast_dev_run)
# STOCKPRED_DRY_RUN=1 python -m scripts.hparam_search

