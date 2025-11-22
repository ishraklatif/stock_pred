<!-- HERO SECTION -->
<h1 align="center">📈 StockPred</h1>
<h3 align="center">Multiseries Temporal Fusion Transformer for ASX Market Forecasting</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Model-TFT-blue" />
  <img src="https://img.shields.io/badge/Framework-PyTorch%20Lightning-orange" />
  <img src="https://img.shields.io/badge/Forecasting-Multiseries-success" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

---

# 🌟 Overview

**StockPred** is a complete research and production-grade time-series forecasting system built on the **Temporal Fusion Transformer (TFT)**.  
It predicts **next-day stock prices for 50+ ASX companies**, using:

- Company OHLCV  
- Global macro indices  
- Commodities & FX  
- Calendar effects  
- Technical indicators  
- News sentiment  
- Sector embeddings  
- Rich multiseries interactions across all tickers  

This repository delivers a fully modular, extensible machine learning pipeline designed for **academic research**, **industry forecasting**, and **portfolio-quality demonstration**.

---

# 🎯 Motivation

Financial forecasting presents challenges such as non-stationarity, regime shifts, multiscale temporal patterns, and multi-asset dependencies. Traditional statistical models cannot fully capture:

- Cross-series relationships  
- Irregular temporal influence  
- High-dimensional feature spaces  
- Long temporal dependencies  

The **Temporal Fusion Transformer (TFT)** addresses these challenges via:

- Gated residual networks  
- Static covariate encoders  
- Variable selection networks  
- Sequence-to-sequence encoder-decoder  
- Multi-head attention  
- Interpretable forecasting  

StockPred demonstrates how TFT can be used in a **real-world financial setting**, producing a robust multiseries forecasting pipeline.

---

# 🧠 Temporal Fusion Transformer (TFT)

## 📚 Architecture

TFT combines recurrent layers, attention mechanisms, gating, and feature selection into a unified interpretable forecasting architecture.

### 📐 High-Level Architecture

```text
                      ┌──────────────────────────────────────┐
                      │          Static Features             │
                      │  (series, sector_id embeddings)      │
                      └──────────────────────────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │   Static Covariate Encoder (GRN)     │
                   └──────────────────────────────────────┘
                                      │
                                      ▼
     ┌──────────────────────────┬──────────────────────────┬─────────────────────────┐
     │ Historical Inputs        │ Known Future Inputs      │ Target Values           │
     │ (indicators, macro,      │ (time_idx, calendar)     │ (close price)           │
     └──────────────────────────┴──────────────────────────┴─────────────────────────┘
                                      │
                                      ▼
                 ┌────────────────────────────────────────────┐
                 │    Variable Selection Networks (VSN)       │
                 └────────────────────────────────────────────┘
                                      │
                                      ▼
           ┌────────────────────────────────┬────────────────────────────────┐
           │         Encoder (GRN + LSTM)   │      Decoder (GRN + LSTM)     │
           └────────────────────────────────┴────────────────────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────────┐
                   │    Multi-Head Temporal Attention         │
                   └──────────────────────────────────────────┘
                                      │
                                      ▼
                     ┌──────────────────────────────────┐
                     │    Quantile Forecast Outputs      │
                     └──────────────────────────────────┘
```

---

# 📁 Project Structure

```text
project/
│
├── config/
│   ├── train_tft.yaml
│   ├── config-search.yaml
│   ├── environment.yaml
│   └── data.yaml
│
├── data/
│   ├── raw_companies/
│   ├── raw_macro/
│   ├── raw_macro_market/
│   ├── processed_companies/
│   ├── processed_macro/
│   ├── processed_macro_market/
│   └── tft_ready_multiseries/
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
│
├── scripts/
│   ├── clean/
│   │   ├── company_clean.py
│   │   ├── macro_clean.py
│   │   └── market_clean.py
│   │
│   ├── compute/
│   │   ├── compute_calendar_features.py
│   │   ├── compute_indicators.py
│   │   └── compute_news_sentiment.py
│   │
│   ├── fetch/
│   │   ├── fetch_company.py
│   │   ├── fetch_macro_main.py
│   │   ├── fetch_macro_market.py
│   │   └── fetch_macro_news.py
│   │
│   ├── merge/
│   │   └── merge_all_data.py
│   │
│   ├── prepare_data_tft.py
│   ├── inspect_data_by_pipeline.py
│   ├── test_tickers.py
│   ├── train.py
│   ├── hparam_search.py
│   └── evaluate_tft.py
│
└── checkpoints_tft/
```

---

# 🔧 Feature Engineering Pipeline

A rich multiseries dataset is constructed using:

- OHLCV signals  
- Calendar effects  
- Macroeconomic indices  
- FX & commodities  
- Technical indicators (140+)  
- Sentiment  
- Sector embeddings  
- Lag-based temporal features  
- Rolling volatility & regimes  

### 🔄 Feature Pipeline Diagram

```text
RAW SOURCES
 ├── Company OHLCV
 ├── Macro Indices
 ├── Market (FX/Commodities)
 ├── News Sentiment
 └── Calendar Files
        │
        ▼
CLEANING
 ├── company_clean.py
 ├── macro_clean.py
 └── market_clean.py
        │
        ▼
FEATURE COMPUTATION
 ├── compute_indicators.py
 ├── compute_calendar_features.py
 └── compute_news_sentiment.py
        │
        ▼
MERGING
 └── merge_all_data.py
        │
        ▼
MULTISERIES PREPARATION
 └── prepare_data_tft.py
        │
        ▼
TRAINING / SEARCH / EVALUATION
 ├── train.py
 ├── hparam_search.py
 └── evaluate_tft.py
```

---

# 🧩 Script Documentation

### **scripts/fetch/**
| Script | Purpose |
|--------|---------|
| `fetch_company.py` | Downloads ASX OHLCV via Yahoo Finance |
| `fetch_macro_main.py` | Global equities (S&P500, FTSE, Nikkei, HSI) |
| `fetch_macro_market.py` | Commodities, FX, metals, DXY |
| `fetch_macro_news.py` | Macro-linked news sentiment feeds |

---

### **scripts/clean/**
| Script | Purpose |
|--------|---------|
| `company_clean.py` | Cleans OHLCV, handles gaps & anomalies |
| `macro_clean.py` | Aligns macro indices, fixes missing observations |
| `market_clean.py` | Normalizes market datasets |

---

### **scripts/compute/**
| Script | Purpose |
|--------|---------|
| `compute_indicators.py` | Computes 140+ TA indicators |
| `compute_calendar_features.py` | AU/US/CN holidays, month/quarter boundaries |
| `compute_news_sentiment.py` | Sentiment scores from news headlines |

---

### **scripts/merge/**
| Script | Purpose |
|--------|---------|
| `merge_all_data.py` | Merges all signals → unified parquet |

---

### **scripts/prepare_data_tft.py**
Creates multiseries dataset with `time_idx`, `series`, `sector_id`, and splits.

---

### **scripts/train.py**
Config-driven TFT training with:

- AdamW  
- Dropout  
- Weight decay  
- EarlyStopping  
- Checkpointing  

---

### **scripts/hparam_search.py**
Grid search over:

- hidden size  
- dropout  
- learning rate  
- weight decay  
- batch size  

Results saved to `hparam_results.csv`.

---

### **scripts/evaluate_tft.py**
Computes RMSE/MAPE:

- Per ticker  
- Per sector  

Outputs CSVs.

---

# 🚀 Usage Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Full Pipeline (Fetch → Clean → Compute → Merge)

```bash
python scripts/pipeline.py
```

---

## 3. Prepare Multiseries Dataset

```bash
python scripts/prepare_data_tft.py
```

---

## 4. Train TFT Model

```bash
python scripts/train.py
```

---

## 5. Run Hyperparameter Search

```bash
python scripts/hparam_search.py
```

Update best config into:

```
config/train_tft.yaml
```

---

## 6. Evaluate Model

```bash
python scripts/evaluate_tft.py
```

---

# 💻 Google Colab Pro Workflow

```bash
!git clone https://github.com/<your_repo>/stock_pred.git
%cd stock_pred
!pip install pytorch-forecasting pytorch-lightning torch pandas numpy
!python scripts/prepare_data_tft.py
!python scripts/train.py
!python scripts/evaluate_tft.py
```

GPU is used automatically.

---

# 📚 References

Lim, B., Arik, S. Ö., et al.  
**Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.**  
NeurIPS 2019.

---

# 🏁 Conclusion

StockPred demonstrates a complete end-to-end multiseries forecasting system powered by the Temporal Fusion Transformer.  
With its modular structure, rich feature engineering pipeline, and research-grade modeling framework, it is suited for both academic research and industry-grade forecasting deployments.
