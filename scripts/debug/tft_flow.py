#!/usr/bin/env python3
"""
Universal TFT flow extractor FOR YOUR EXACT DATA FORMAT.

Your DataLoader returns a TUPLE:
    raw_batch = (batch_x_dict, ?, ?)

Where the TRUE TARGET is inside:
    batch_x["decoder_target"]

This script extracts correctly and outputs a CSV.
"""

import pandas as pd
import torch
import csv
from pytorch_forecasting import TimeSeriesDataSet

# =========================================================
# 1. Load data
# =========================================================
PATH = "data/tft_ready_multiseries/train.parquet"
df = pd.read_parquet(PATH)

print("\n=== RAW TRAIN SAMPLE ===")
print(df.head())


# =========================================================
# 2. Build features
# =========================================================
TARGET = "close"
TIME_IDX = "time_idx"
GROUP_ID = "series"

static_categoricals = ["series", "sector_id"]
known_future_reals = ["time_idx"]

unknown_reals = ["close"] + [
    c for c in df.columns
    if c not in ["Date", GROUP_ID, TIME_IDX, "sector_id"]
       and df[c].dtype != "object"
       and c != "close"
]

training = TimeSeriesDataSet(
    df,
    time_idx=TIME_IDX,
    target=TARGET,
    group_ids=[GROUP_ID],
    max_encoder_length=60,
    max_prediction_length=5,
    static_categoricals=static_categoricals,
    static_reals=[],
    time_varying_known_reals=known_future_reals,
    time_varying_unknown_reals=unknown_reals,
)

print("\n=== TFT DATASET CREATED ===")
print(training)


# =========================================================
# 3. Extract ONE batch (YOUR FORMAT)
# =========================================================
loader = training.to_dataloader(train=True, batch_size=4)

print("\n[INFO] Extracting batch...\n")

raw_batch = next(iter(loader))

# Your format: TUPLE, batch_x is FIRST element
if isinstance(raw_batch, tuple):
    batch_x = raw_batch[0]

elif isinstance(raw_batch, dict):
    batch_x = raw_batch

else:
    raise ValueError(f"Unknown batch type: {type(raw_batch)}")


# =========================================================
# 4. Extract TARGET from inside batch_x
# =========================================================
if "decoder_target" not in batch_x:
    print("\nAvailable keys:", batch_x.keys())
    raise ValueError("decoder_target NOT FOUND inside batch_x — unexpected!")

batch_y = batch_x["decoder_target"]


# =========================================================
# 5. Print shapes
# =========================================================
print("\n=== BATCH_X KEYS ===")
print(list(batch_x.keys()))

print("\n=== TARGET (decoder_target) SHAPE ===")
print(batch_y.shape)


# =========================================================
# 6. Save CSV
# =========================================================
rows = []
for k, v in batch_x.items():
    if torch.is_tensor(v):
        rows.append([k, list(v.shape)])
    else:
        rows.append([k, str(type(v))])

rows.append(["target(decoder_target)", list(batch_y.shape)])

with open("tft_flow_output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["component", "shape"])
    writer.writerows(rows)

print("\n[INFO] Saved → tft_flow_output.csv")

print("""
=============== TFT FLOW (YOUR VERSION) ===============
train.parquet →
  TimeSeriesDataSet →
    DataLoader returns TUPLE:
        raw_batch = (batch_x_dict, extra1, extra2)
    batch_x = raw_batch[0]
    decoder_target = the TRUE target tensor

Correctly extracted.
=======================================================
""")
