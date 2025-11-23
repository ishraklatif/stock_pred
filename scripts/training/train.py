#!/usr/bin/env python3
"""
Config-driven TFT Training (Clean + Consistent)
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")
DRY_RUN = os.getenv("STOCKPRED_DRY_RUN", "0") == "1"


class MyTFT(TemporalFusionTransformer):
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.optimizer_params["weight_decay"],
        )


def load_config(path="config/train_tft.yaml"):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------
# Clean NaN/inf → 0
# --------------------------------
def clean_numeric(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# --------------------------------
# Dataset Builder
# --------------------------------
def build_datasets(cfg: Dict[str, Any], train_df, val_df):

    train_df = clean_numeric(train_df)
    val_df   = clean_numeric(val_df)

    data_cfg = cfg["data"]
    target = data_cfg["target"]
    time_idx = data_cfg["time_idx"]
    group_id = data_cfg["group_id"]

    # categorical static
    for col in data_cfg["static_categoricals"]:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
            val_df[col] = val_df[col].astype(str)

    ignore = {"Date", group_id, time_idx, target, "sector_id"}
    numeric_feats = [
        c for c in train_df.columns
        if c not in ignore and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    print(f"[INFO] Final feature count = {len(numeric_feats)}")

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
        training, val_df, predict=False, stop_randomization=True
    )

    return training, validation, loss, output_size


# --------------------------------
# Main
# --------------------------------
def main():
    cfg = load_config()
    seed_everything(cfg["training"]["seed"], workers=True)

    train_df = pd.read_parquet(cfg["paths"]["data_dir"] + "/train.parquet")
    val_df   = pd.read_parquet(cfg["paths"]["data_dir"] + "/val.parquet")

    training, validation, loss, output_size = build_datasets(cfg, train_df, val_df)

    train_loader = training.to_dataloader(
        train=True,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )

    val_loader = validation.to_dataloader(
        train=False,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )

    ckpt_last = cfg["paths"]["model_dir"] + "/last.ckpt"
    os.makedirs(cfg["paths"]["model_dir"], exist_ok=True)

    tft_kwargs = dict(
        learning_rate=cfg["model"]["learning_rate"],
        hidden_size=cfg["model"]["hidden_size"],
        hidden_continuous_size=cfg["model"]["hidden_continuous_size"],
        attention_head_size=cfg["model"]["attention_head_size"],
        dropout=cfg["model"]["dropout"],
        optimizer="adamw",
        optimizer_params={"weight_decay": cfg["model"]["weight_decay"]},
        loss=loss,
        output_size=output_size,
    )

    if os.path.exists(ckpt_last) and not DRY_RUN:
        print(f"[INFO] Resuming from checkpoint… {ckpt_last}")
        model = MyTFT.from_dataset(training, **tft_kwargs)
        ckpt = torch.load(ckpt_last, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model = MyTFT.from_dataset(training, **tft_kwargs)

    trainer = Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=cfg["paths"]["model_dir"],
                filename="tft",
                save_last=True,
                save_top_k=1,
                monitor="val_loss"
            ),
            EarlyStopping(monitor="val_loss", patience=cfg["optim"]["patience"])
        ],
        gradient_clip_val=cfg["optim"]["gradient_clip_val"],
        log_every_n_steps=cfg["training"]["log_every"],
        fast_dev_run=1 if DRY_RUN else False,
    )

    trainer.fit(model, train_loader, val_loader)

    if not DRY_RUN:
        trainer.save_checkpoint(ckpt_last)


if __name__ == "__main__":
    main()
