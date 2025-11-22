#!/usr/bin/env python3
"""
check_project_integrity.py

Comprehensive integrity validator for STOCKPRED project.
Ensures all components (YAML configs, parquet structure, dataset splits,
training scripts, evaluation, and hyperparameter search config)
are consistent and ready for training.

Author: Ishrak Latif + ChatGPT
"""

import os
import sys
import importlib
import traceback
import pandas as pd
import yaml
import numpy as np

# Add project root so scripts/ becomes importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



# =====================================================================
# Helper Functions
# =====================================================================

def header(title):
    print(f"\n==== {title} ====")


def fail(msg, err=None):
    print("\n--- ERROR DETAILS ---")
    if err:
        traceback.print_exc()
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def ok(msg="OK"):
    print(f"[OK] {msg}")


def try_step(name, fn):
    header(name)
    try:
        fn()
        ok(name)
    except Exception as e:
        fail(name, e)


# =====================================================================
# 1. IMPORT TEST
# =====================================================================

def step_check_imports():
    required = [
        "pandas",
        "torch",
        "pytorch_forecasting",
        "lightning",
        "yaml"
    ]
    for module in required:
        importlib.import_module(module)


# =====================================================================
# 2. VALIDATE YAML CONFIGS
# =====================================================================

def validate_yaml_structure(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML file missing: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"{path} must load a dict")
    return cfg


def step_yaml_configs():
    cfg_train = validate_yaml_structure("config/train_tft.yaml")
    cfg_search = validate_yaml_structure("config/config-search.yaml")
    cfg_data = validate_yaml_structure("config/data.yaml")

    # Check specific sections exist
    if "search_space" not in cfg_search:
        raise ValueError("config-search.yaml must contain 'search_space:'")
    if not isinstance(cfg_search["search_space"], dict):
        raise ValueError("'search_space' in config-search.yaml must be a dict")

    # Validate search-space entries
    for key, val in cfg_search["search_space"].items():
        if not isinstance(val, list):
            raise ValueError(f"search_space['{key}'] must be a list")

    # Basic keys for train_tft
    required_train_keys = ["data", "model", "paths", "training", "optim", "system"]
    for key in required_train_keys:
        if key not in cfg_train:
            raise ValueError(f"Missing key '{key}' in train_tft.yaml")


# =====================================================================
# 3. PARQUET FILE PRESENCE
# =====================================================================

def step_parquet_files():
    base = "data/tft_ready_multiseries"
    for file in ["train.parquet", "val.parquet", "test.parquet"]:
        path = f"{base}/{file}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")


# =====================================================================
# 4. REQUIRED TFT COLUMNS
# =====================================================================

def step_required_columns():
    df = pd.read_parquet("data/tft_ready_multiseries/train.parquet")
    required = ["Date", "series", "sector_id", "time_idx", "close"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")


# =====================================================================
# 5. SECTOR ID CHECK
# =====================================================================

def step_sector_id_check():
    df = pd.read_parquet("data/tft_ready_multiseries/train.parquet")
    missing = df[df["sector_id"] == "Unknown"]["series"].unique().tolist()
    if missing:
        print("[WARN] Unknown sector ID for:", missing)
    else:
        print("[INFO] All tickers have sector_id")


# =====================================================================
# 6. PREVENT INVALID COLUMN NAMES (dot, hyphen)
# =====================================================================

def step_invalid_columns():
    df = pd.read_parquet("data/tft_ready_multiseries/train.parquet")
    bad = [c for c in df.columns if "." in c or "-" in c]
    if bad:
        raise ValueError(f"Invalid column names (dots or hyphens not allowed): {bad}")


# =====================================================================
# 7. DATASET BUILD CHECK (train.py)
# =====================================================================

def step_test_build_datasets():
    from scripts.train import load_config, build_datasets

    cfg = load_config("config/train_tft.yaml")

    train_df = pd.read_parquet("data/tft_ready_multiseries/train.parquet")
    val_df   = pd.read_parquet("data/tft_ready_multiseries/val.parquet")

    training, validation, loss, output = build_datasets(cfg, train_df, val_df)

    sample = next(iter(training.to_dataloader(train=True, batch_size=4)))
    x, y = sample
    if x is None or y is None:
        raise ValueError("Dataset produced empty batches")


# =====================================================================
# 8. MODEL INITIALIZATION (train.py)
# =====================================================================

def step_model_initialization():
    from scripts.train import load_config, build_datasets, MyTFT
    cfg = load_config("config/train_tft.yaml")

    train_df = pd.read_parquet("data/tft_ready_multiseries/train.parquet")
    val_df   = pd.read_parquet("data/tft_ready_multiseries/val.parquet")

    training, validation, loss, output = build_datasets(cfg, train_df, val_df)
    model = MyTFT.from_dataset(
        training,
        hidden_size=cfg["model"]["hidden_size"],
        attention_head_size=cfg["model"]["attention_head_size"],
        dropout=cfg["model"]["dropout"],
        learning_rate=cfg["model"]["learning_rate"],
        loss=loss,
        output_size=output,
        optimizer="adamw",
        optimizer_params={"weight_decay": cfg["model"]["weight_decay"]},
    )


# =====================================================================
# 9. VALIDATE HYPERPARAM SEARCH CONFIG
# =====================================================================

def step_validate_search_config():
    from scripts.hparam_search import load_search_config
    search_space, data_cfg = load_search_config()

    if not isinstance(search_space, dict):
        raise ValueError("'search_space' must be dict")

    for k, v in search_space.items():
        if not isinstance(v, list):
            raise ValueError(f"search_space['{k}'] must be a list")

    print("[INFO] search_space keys:", list(search_space.keys()))



# =====================================================================
# MAIN
# =====================================================================

def main():
    print("===============================================")
    print("🚀 STOCKPRED PROJECT INTEGRITY CHECK")
    print("===============================================\n")

    try_step("Imports", step_check_imports)
    try_step("YAML Configs Valid", step_yaml_configs)
    try_step("Parquet Files Present", step_parquet_files)
    try_step("Required TFT Columns", step_required_columns)
    try_step("sector_id Check", step_sector_id_check)
    try_step("Invalid Column Name Check", step_invalid_columns)
    try_step("Dataset Construction (train.py)", step_test_build_datasets)
    try_step("Model Initialization (train.py)", step_model_initialization)
    try_step("Hyperparameter Search Config", step_validate_search_config)

    print("\n===============================================")
    print("🎉 ALL CHECKS PASSED — PROJECT IS READY TO TRAIN")
    print("===============================================")


if __name__ == "__main__":
    main()
