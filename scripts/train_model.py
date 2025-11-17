# #!/usr/bin/env python3
# """
# train_tft_multiseries.py

# Train a single multi-series Temporal Fusion Transformer (TFT) model
# using combined train/val parquet files:

#     data/tft_ready_multiseries/train.parquet
#     data/tft_ready_multiseries/val.parquet

# All companies are trained together, using `series` as group_id.
# """

# import os
# import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")

# import torch
# from lightning.pytorch import Trainer, seed_everything
# from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
# from pytorch_forecasting.metrics import QuantileLoss
# from pytorch_forecasting.data import GroupNormalizer

# # =============================================================================
# # CONFIG
# # =============================================================================
# DATA_DIR = "data/tft_ready_multiseries"
# MODEL_DIR = "models_tft_multiseries"

# TARGET = "close"
# TIME_IDX = "time_idx"
# GROUP_ID = "series"

# MAX_ENCODER_LENGTH = 60
# MAX_PREDICTION_LENGTH = 5

# BATCH_SIZE = 64
# EPOCHS = 50
# SEED = 42
# NUM_WORKERS = 0  # safer for Windows


# # =============================================================================
# # DEVICE SELECTION
# # =============================================================================
# def create_trainer():
#     """Auto-select GPU/MPS/CPU + add callbacks."""

#     if torch.backends.mps.is_available():
#         accelerator = "mps"
#         devices = 1
#         print("[INFO] Using Apple Silicon (MPS)")
#     elif torch.cuda.is_available():
#         accelerator = "gpu"
#         devices = 1
#         print("[INFO] Using CUDA GPU")
#     else:
#         accelerator = "cpu"
#         devices = 1
#         print("[INFO] Using CPU")

#     early_stop = EarlyStopping(
#         monitor="val_loss",
#         patience=5,
#         mode="min",
#         verbose=True,
#     )

#     checkpoint_cb = ModelCheckpoint(
#         dirpath=MODEL_DIR,
#         filename="tft-{epoch:02d}-{val_loss:.4f}",
#         save_top_k=1,
#         save_last=True,
#         monitor="val_loss",
#         mode="min",
#     )

#     trainer = Trainer(
#         max_epochs=EPOCHS,
#         accelerator=accelerator,
#         devices=devices,
#         gradient_clip_val=0.1,
#         log_every_n_steps=10,
#         default_root_dir=MODEL_DIR,
#         callbacks=[early_stop, checkpoint_cb],
#     )

#     return trainer


# # =============================================================================
# # BUILD DATASETS
# # =============================================================================
# def build_datasets(train_df, val_df):
#     """Build PyTorch Forecasting TimeSeriesDataSet objects."""

#     # --- Feature selection ---
#     feature_cols = [
#         c for c in train_df.columns
#         if c not in ["Date", GROUP_ID, TIME_IDX, TARGET]
#         and pd.api.types.is_numeric_dtype(train_df[c])
#     ]

#     print(f"[INFO] Numeric features used ({len(feature_cols)}):")
#     for f in feature_cols:
#         print(f"   - {f}")

#     loss = QuantileLoss()
#     output_size = len(loss.quantiles)

#     # --- Training dataset ---
#     training = TimeSeriesDataSet(
#         train_df,
#         time_idx=TIME_IDX,
#         target=TARGET,
#         group_ids=[GROUP_ID],
#         min_encoder_length=MAX_ENCODER_LENGTH,
#         max_encoder_length=MAX_ENCODER_LENGTH,
#         min_prediction_length=MAX_PREDICTION_LENGTH,
#         max_prediction_length=MAX_PREDICTION_LENGTH,
#         static_categoricals=[GROUP_ID],
#         time_varying_unknown_reals=[TARGET] + feature_cols,
#         time_varying_known_reals=[TIME_IDX],
#         target_normalizer=GroupNormalizer(groups=[GROUP_ID]),
#     )

#     # --- Validation dataset ---
#     validation = training.from_dataset(
#         training,
#         val_df,
#         predict=True,
#         stop_randomization=True,
#     )

#     return training, validation, loss, output_size


# # =============================================================================
# # MAIN
# # =============================================================================
# def main():
#     seed_everything(SEED, workers=True)

#     print("[INFO] Loading multi-series dataset...")
#     train_path = os.path.join(DATA_DIR, "train.parquet")
#     val_path = os.path.join(DATA_DIR, "val.parquet")

#     train_df = pd.read_parquet(train_path)
#     val_df = pd.read_parquet(val_path)

#     print(f"[INFO] Train rows: {len(train_df):,}")
#     print(f"[INFO] Val rows:   {len(val_df):,}")
#     print(f"[INFO] Companies:  {train_df['series'].nunique()} found")

#     # Build datasets
#     training, validation, loss, output_size = build_datasets(train_df, val_df)

#     # Dataloaders
#     train_loader = training.to_dataloader(
#         train=True,
#         batch_size=BATCH_SIZE,
#         num_workers=NUM_WORKERS,
#     )
#     val_loader = validation.to_dataloader(
#         train=False,
#         batch_size=BATCH_SIZE,
#         num_workers=NUM_WORKERS,
#     )

#     # Model directory
#     os.makedirs(MODEL_DIR, exist_ok=True)

#     # Build TFT
#     tft = TemporalFusionTransformer.from_dataset(
#         training,
#         learning_rate=1e-3,
#         hidden_size=32,
#         attention_head_size=4,
#         dropout=0.1,
#         hidden_continuous_size=16,
#         loss=loss,
#         output_size=output_size,
#         log_interval=10,
#         log_val_interval=1,
#     )

#     trainer = create_trainer()

#     print("[INFO] Training multi-series TFT...")
#     trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

#     trainer.save_checkpoint(os.path.join(MODEL_DIR, "last.ckpt"))
#     print(f"\n[DONE] Multi-series TFT model trained and saved to: {MODEL_DIR}/last.ckpt")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
Resumable TFT Training Script for Google Colab
- Trains only 2 epochs at a time
- Auto-saves checkpoint
- Automatically resumes from last checkpoint if available
"""

import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer


# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = "data/tft_ready_multiseries"
MODEL_DIR = "checkpoints_tft"

TARGET = "close"
TIME_IDX = "time_idx"
GROUP_ID = "series"

MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 5

BATCH_SIZE = 64
EPOCHS_PER_RUN = 50      # << train only 2 epochs each session
SEED = 42
NUM_WORKERS = 0


# =============================================================================
# DEVICE SELECTION
# =============================================================================
def select_device():
    """Auto-select GPU/CPU."""
    if torch.cuda.is_available():
        print("[INFO] Using CUDA GPU")
        return "gpu", 1
    else:
        print("[INFO] Using CPU")
        return "cpu", 1


# =============================================================================
# BUILD DATASET
# =============================================================================
def build_datasets(train_df, val_df):
    feature_cols = [
        c for c in train_df.columns
        if c not in ["Date", GROUP_ID, TIME_IDX, TARGET]
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    print(f"[INFO] Using {len(feature_cols)} numerical features.")

    loss = QuantileLoss()
    output_size = len(loss.quantiles)

    training = TimeSeriesDataSet(
        train_df,
        time_idx=TIME_IDX,
        target=TARGET,
        group_ids=[GROUP_ID],
        min_encoder_length=MAX_ENCODER_LENGTH,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=MAX_PREDICTION_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=[GROUP_ID],
        time_varying_unknown_reals=[TARGET] + feature_cols,
        time_varying_known_reals=[TIME_IDX],
        target_normalizer=GroupNormalizer(groups=[GROUP_ID]),
    )

    validation = training.from_dataset(
        training,
        val_df,
        predict=True,
        stop_randomization=True,
    )

    return training, validation, loss, output_size


# =============================================================================
# MAIN
# =============================================================================
def main():
    seed_everything(SEED, workers=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("[INFO] Loading dataset...")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))

    # build datasets
    training, validation, loss, output_size = build_datasets(train_df, val_df)

    train_loader = training.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # Checkpoint callback
    checkpoint_cb = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename="tft",
        save_last=True,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    accelerator, devices = select_device()

    # Trainer (only 2 epochs)
    trainer = Trainer(
        max_epochs=EPOCHS_PER_RUN,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=MODEL_DIR,
        callbacks=[checkpoint_cb],
        gradient_clip_val=0.1,
        log_every_n_steps=10,
    )

    checkpoint_path = os.path.join(MODEL_DIR, "last.ckpt")

    if os.path.exists(checkpoint_path):
        print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")

        # Rebuild model architecture
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=1e-3,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            loss=loss,
            output_size=output_size,
        )

        # Load stored weights
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        tft.load_state_dict(ckpt["state_dict"])
        print("[INFO] Weights loaded and model restored.")

    else:
        print("[INFO] Starting new training session...")
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=1e-3,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            loss=loss,
            output_size=output_size,
        )


    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.save_checkpoint(checkpoint_path)
    print("\n[INFO] Training complete. Checkpoint saved.")


if __name__ == "__main__":
    main()
