#!/usr/bin/env python3
"""
Config-driven TFT Training Script
---------------------------------

Reads hyperparameters from config/train_tft.yaml
Trains a small, anti-overfitting TFT on ASX multiseries dataset.

Features:
- Config-driven (paths, model, training, optim, system)
- AdamW optimizer via a custom TFT subclass
- Robust handling of:
    * bad column names ('.' and '-' → '_')
    * non-finite values (NaN / inf) in numeric features
- Optional DRY RUN via env var:
    STOCKPRED_DRY_RUN=1 python -m scripts.train

Author: Ishrak + ChatGPT
"""

import os
import warnings
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")


# =====================================================================
# ENV / GLOBALS
# =====================================================================
DRY_RUN = os.getenv("STOCKPRED_DRY_RUN", "0") == "1"


# =====================================================================
# CUSTOM OPTIMIZER (AdamW)
# =====================================================================
class MyTFT(TemporalFusionTransformer):
    """
    TFT subclass that uses AdamW with weight decay from hparams.optimizer_params.
    """

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt_params = getattr(self.hparams, "optimizer_params", {}) or {}
        weight_decay = opt_params.get("weight_decay", 0.0)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        return optimizer


# =====================================================================
# CONFIG
# =====================================================================
def load_config(path: str = "config/train_tft.yaml") -> Dict[str, Any]:
    import yaml

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"[INFO] Loaded config from {path}")
    return cfg


# =====================================================================
# DEVICE SELECTION
# =====================================================================
def select_device(cfg: Dict[str, Any]) -> Tuple[str, int]:
    use_gpu = cfg.get("system", {}).get("use_gpu_if_available", True)
    if use_gpu and torch.cuda.is_available():
        print("[INFO] Using CUDA GPU")
        return "gpu", 1
    print("[INFO] Using CPU")
    return "cpu", 1


# =====================================================================
# SANITISATION HELPERS
# =====================================================================
def sanitize_column_names(train_df: pd.DataFrame,
                          val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replace '.' and '-' in column names with '_' so that PyTorch Forecasting
    (and TFT) are happy.
    """
    rename_map = {c: c.replace(".", "_").replace("-", "_") for c in train_df.columns}
    if any(new != old for old, new in rename_map.items()):
        print("[INFO] Sanitising column names ('.' / '-' → '_').")
        train_df = train_df.rename(columns=rename_map)
        val_df = val_df.rename(columns=rename_map)
    return train_df, val_df


def drop_nonfinite_rows(
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    target: str, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    - Drop any numeric feature columns that contain NaN/inf in *either* split.
    - Then drop rows with non-finite values in target + remaining features.
    """
    good_features = []
    dropped_features = []

    for col in feature_cols:
        train_vals = train_df[col].to_numpy()
        val_vals = val_df[col].to_numpy()
        if np.isfinite(train_vals).all() and np.isfinite(val_vals).all():
            good_features.append(col)
        else:
            dropped_features.append(col)

    if dropped_features:
        print(f"[WARN] Dropping {len(dropped_features)} features with non-finite values:")
        for c in dropped_features[:20]:
            print("   -", c)
        if len(dropped_features) > 20:
            print(f"   ... and {len(dropped_features) - 20} more")

    feature_cols = good_features

    def _clean_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
        cols = [target] + feature_cols
        arr = np.column_stack([df[c].to_numpy() for c in cols])
        mask = np.isfinite(arr).all(axis=1)
        before = len(df)
        df = df.loc[mask].copy()
        removed = before - len(df)
        if removed > 0:
            print(f"[INFO] Dropped {removed} rows with non-finite values in {name}.")
        return df

    train_df = _clean_df(train_df, "train")
    val_df = _clean_df(val_df, "val")

    return train_df, val_df, feature_cols


# =====================================================================
# BUILD DATASETS
# =====================================================================
def build_datasets(cfg: Dict[str, Any],
                   train_df: pd.DataFrame,
                   val_df: pd.DataFrame):
    data_cfg = cfg["data"]
    target = data_cfg["target"]
    time_idx = data_cfg["time_idx"]
    group_id = data_cfg["group_id"]

    # Ensure categorical static cols are strings
    for col in data_cfg.get("static_categoricals", []):
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
        if col in val_df.columns:
            val_df[col] = val_df[col].astype(str)

    # Column name sanitisation for dots / hyphens
    train_df, val_df = sanitize_column_names(train_df, val_df)

    # Infer numeric observed features
    ignore_cols = set(
        ["Date", group_id, time_idx, target] +
        data_cfg.get("static_categoricals", [])
    )
    feature_cols = [
        c for c in train_df.columns
        if c not in ignore_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    print(f"[INFO] Initial numeric feature count = {len(feature_cols)}")

    # Drop columns/rows with NaN / inf
    train_df, val_df, feature_cols = drop_nonfinite_rows(
        train_df, val_df, target, feature_cols
    )

    print(f"[INFO] Final observed historical features = {len(feature_cols)}")

    # Loss
    loss = QuantileLoss()
    output_size = len(loss.quantiles)

    # TimeSeriesDataSet — training
    training = TimeSeriesDataSet(
        train_df,
        time_idx=time_idx,
        target=target,
        group_ids=[group_id],

        max_encoder_length=data_cfg["max_encoder_length"],
        max_prediction_length=data_cfg["max_prediction_length"],

        static_categoricals=data_cfg.get("static_categoricals", []),
        static_reals=[],

        time_varying_known_reals=data_cfg.get("known_future_reals", [time_idx]),
        time_varying_known_categoricals=[],

        time_varying_unknown_reals=[target] + feature_cols,
        time_varying_unknown_categoricals=[],
    )

    # Validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_df,
        predict=False,
        stop_randomization=True,
    )

    return training, validation, loss, output_size


# =====================================================================
# MAIN
# =====================================================================
def main():
    cfg = load_config()

    seed_everything(cfg["training"].get("seed", 42), workers=True)

    data_dir = cfg["paths"]["data_dir"]
    model_dir = cfg["paths"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    train_path = os.path.join(data_dir, "train.parquet")
    val_path = os.path.join(data_dir, "val.parquet")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    print(f"[INFO] Train shape: {train_df.shape}")
    print(f"[INFO] Val   shape: {val_df.shape}")

    # Build datasets
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

    # Callbacks
    ckpt_last = os.path.join(model_dir, "last.ckpt")

    checkpoint_cb = ModelCheckpoint(
        dirpath=model_dir,
        filename="tft",
        save_last=True,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=cfg["optim"]["patience"],
        mode="min",
        verbose=True,
    )

    # Device
    accelerator, devices = select_device(cfg)

    # TFT kwargs
    model_cfg = cfg["model"]
    tft_kwargs = dict(
        learning_rate=model_cfg["learning_rate"],
        hidden_size=model_cfg["hidden_size"],
        hidden_continuous_size=model_cfg["hidden_continuous_size"],
        attention_head_size=model_cfg["attention_head_size"],
        dropout=model_cfg["dropout"],
        loss=loss,
        output_size=output_size,
        optimizer="adamw",
        optimizer_params={"weight_decay": model_cfg["weight_decay"]},
    )

    # Trainer
    trainer = Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator=accelerator,
        devices=devices,
        default_root_dir=model_dir,
        callbacks=[checkpoint_cb, early_stop_cb],
        gradient_clip_val=cfg["optim"]["gradient_clip_val"],
        log_every_n_steps=cfg["training"]["log_every"],
        fast_dev_run=1 if DRY_RUN else False,
    )

    # Resume or start new
    if os.path.exists(ckpt_last) and not DRY_RUN:
        print(f"[INFO] Resuming from: {ckpt_last}")
        tft = MyTFT.from_dataset(training, **tft_kwargs)
        ckpt = torch.load(ckpt_last, map_location="cpu")
        tft.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        if DRY_RUN:
            print("[INFO] DRY RUN mode → always starting fresh model.")
        else:
            print("[INFO] Starting new training run...")
        tft = MyTFT.from_dataset(training, **tft_kwargs)

    # Fit
    trainer.fit(tft, train_loader, val_loader)

    if not DRY_RUN:
        trainer.save_checkpoint(ckpt_last)
        print(f"[INFO] Training complete. Saved to: {ckpt_last}")
    else:
        print("[INFO] DRY RUN complete (no checkpoints saved).")


if __name__ == "__main__":
    main()
