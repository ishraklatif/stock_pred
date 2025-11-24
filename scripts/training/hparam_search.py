#!/usr/bin/env python3
"""
Resumable Hyperparameter Search for TFT (Grid Search)
----------------------------------------------------

Features:
✔ Fully resumable (safe for Google Colab)
✔ Tracks completed trials in hparam_progress.csv
✔ Saves results after each trial → hparam_results.csv
✔ Skips finished trials on rerun
✔ Optional per-session limit (MAX_TRIALS_PER_SESSION)
✔ Anti-overfitting TFT configuration
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
VAL_PATH   = f"{DATA_DIR}/val.parquet"

# --------------------------------------------------
# Optional: limit trials per run (safe for Colab)
# --------------------------------------------------
MAX_TRIALS_PER_SESSION = 4    # set None to disable


# ==================================================
# Custom TFT (AdamW optimizer)
# ==================================================
class MyTFT(TemporalFusionTransformer):
    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        wd = self.hparams.optimizer_params["weight_decay"]
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)


# ==================================================
# Config loader
# ==================================================
def load_search_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg["search_space"], cfg["data"]


# ==================================================
# Clean numeric columns
# ==================================================
def clean_numeric(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ==================================================
# Dataset builder
# ==================================================
def build_datasets(train_df, val_df, data_cfg):

    train_df = clean_numeric(train_df)
    val_df = clean_numeric(val_df)

    # categorical casting
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


# ==================================================
# Run one hyperparameter trial
# ==================================================
def run_trial(hparams, data_cfg, trial_id):

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

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
            EarlyStopping(monitor="val_loss", patience=5),
        ],
        log_every_n_steps=50,
    )

    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics["val_loss"].item()


# ==================================================
# Progress logging (resumable search)
# ==================================================
def load_completed_trials(path="hparam_progress.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path)
        return set(df[df.status == "done"].trial_id)
    return set()


def save_trial_status(trial_id, status, path="hparam_progress.csv"):
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["trial_id", "status"])

    df = pd.concat([df, pd.DataFrame([{"trial_id": trial_id, "status": status}])])
    df.to_csv(path, index=False)


# ==================================================
# Main
# ==================================================
def main():

    seed_everything(42)

    search_space, data_cfg = load_search_config()
    keys   = list(search_space.keys())
    values = list(search_space.values())
    combos = list(itertools.product(*values))

    print(f"[INFO] Total combinations = {len(combos)}")

    # load completed runs
    completed = load_completed_trials()
    print(f"[INFO] Completed trials: {sorted(list(completed))}")

    results = []
    session_count = 0

    # loop over all combinations
    for i, combo in enumerate(combos):

        # skip previously finished trials
        if i in completed:
            print(f"[SKIP] Trial {i} already completed.")
            continue

        # enforce session limit
        if MAX_TRIALS_PER_SESSION is not None and session_count >= MAX_TRIALS_PER_SESSION:
            print("\n[INFO] Session trial limit reached. Stop now.")
            break

        hparams = dict(zip(keys, combo))
        hparams["max_epochs"] = search_space["max_epochs"][0]

        print(f"\n[TRIAL {i}] {hparams}")

        val_loss = run_trial(hparams, data_cfg, i)

        # save results
        results.append({**hparams, "val_loss": val_loss})
        pd.DataFrame(results).to_csv("hparam_results.csv", index=False)

        # save progress
        save_trial_status(i, "done")

        session_count += 1

    print("\n[INFO] DONE → partial or full results saved.")


if __name__ == "__main__":
    main()
