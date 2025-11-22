<!-- HERO SECTION -->
<h1 align="center">📈 StockPred</h1>
<h3 align="center">Multiseries PatchTST-based ASX Market Forecasting</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Model-PatchTST-blue" />
  <img src="https://img.shields.io/badge/Framework-PyTorch%20Lightning-orange" />
  <img src="https://img.shields.io/badge/Forecasting-Multiseries-success" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

---

# 🌟 Overview

**StockPred** is a research-grade and production-oriented multiseries forecasting system built on the state-of-the-art **PatchTST (Time-Series Patch Transformer)** architecture.

It predicts **next-day stock prices for ASX companies** using a rich blend of:

- Company OHLCV  
- Global macro indicators  
- Commodities & FX  
- Calendar & holiday effects  
- 140+ technical indicators  
- News sentiment  
- Sector embeddings  
- Cross-series temporal dependencies  

The project delivers a fully modular, extensible forecasting pipeline for **academic research**, **industry forecasting**, and **portfolio-quality AI engineering**.

---

# 🎯 Motivation

Financial forecasting demands models capable of handling:

- Multiseries correlations  
- Volatile and non-stationary regimes  
- High-dimensional feature spaces (1200+ engineered signals)  
- Multi-horizon forecasting  
- Irregular and long-range temporal structure  

Classical models fail at scale.  
Transformers solve this — but standard attention scales poorly with very high feature counts.

**PatchTST** offers the advantages of Transformers without the feature explosion:

- Tokenization over *time patches* (not over features)  
- Shared encoders across features → prevents overfitting  
- Excellent stability with 1000+ time-varying features  
- SOTA performance across forecasting benchmarks  

StockPred leverages PatchTST to achieve stable and scalable forecasting performance across 20–100 tickers.

---

# 🧠 PatchTST Architecture

PatchTST transforms time-series segments into patch embeddings and processes them with a transformer encoder.

### 📐 High-Level Architecture

```text
                    ┌────────────────────────────────────┐
                    │   Raw Time Series Segments         │
                    │ (per feature, per time window)     │
                    └────────────────────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────────┐
                    │           Patch Extraction          │
                    │   (e.g., length=32, stride=16)      │
                    └────────────────────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────────┐
                    │   Patch Embedding (Shared Encoder) │
                    └────────────────────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────────┐
                    │     Transformer Encoder Layers     │
                    │   (Self-Attention + FFN blocks)    │
                    └────────────────────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────────┐
                    │     Multi-horizon Predictions      │
                    └────────────────────────────────────┘
```

### Why PatchTST for StockPred?

- Scales easily to **1200+ numerical features**  
- Robust for **multiseries forecasting**  
- Outperforms temporal CNNs, TFT, and Informer in many benchmarks  
- Stable even with small prediction windows (e.g., 1–5 days)  
- Avoids feature-selection overhead of TFT  

---

# 📁 Project Structure

```text
project/
│
├── config/
│   ├── train_patchtst.yaml
│   ├── config-search-patchtst.yaml
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
│   │
│   ├── train_patchtst.py
│   ├── hparam_search_patchtst.py
│   └── evaluate_patchtst.py
│
└── checkpoints_patchtst/
```

---

# 🔧 Feature Engineering Pipeline

StockPred constructs a **rich, high-dimensional multiseries dataset** from:

- Company OHLCV  
- Macro indices (S&P500, FTSE, Nikkei, etc.)  
- Market-level signals (DXY, Gold, Brent, AUD/USD, VIX…)  
- 140+ technical indicators  
- Calendar seasonalities  
- Business days, holidays, month-end/quarter-end  
- News sentiment via FinBERT  
- Sector embeddings  

---

# 🚀 Usage Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare Dataset

```bash
python scripts/prepare_data_tft.py
```

## 3. Train PatchTST Model

```bash
python scripts/train_patchtst.py
```

## 4. Run Hyperparameter Search

```bash
python scripts/hparam_search_patchtst.py
```

## 5. Evaluate the Model

```bash
python scripts/evaluate_patchtst.py
```

---

# 📚 References

- Nie, Y. et al. **"Time Series Patching Transformer"**, NeurIPS 2023  
- Zerveas, G. et al. **"A Transformer-based Framework for Multivariate Time Series Representation Learning"**, ICLR 2021  
- Lim, B., Arik, S. Ö. **"Temporal Fusion Transformers"**, NeurIPS 2019  

---

# 🏁 Conclusion

StockPred now leverages **PatchTST**, enabling scalable, stable multiseries forecasting across high-dimensional datasets.  
This modernized pipeline is suitable for academic research and industry-level forecasting deployments.
