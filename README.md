# WTI Crude Oil Direction Classification

**NYU VIP — ML in Energy Commodities (Spring 2026)**

Binary next-day price direction prediction (↑ / ↓) for WTI crude oil using five model families, three investment strategies, and two alternative data sources. All models are evaluated strictly out-of-sample via time-series cross-validation — no data leakage, no shuffling.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Notebooks & Status](#notebooks--status)
4. [Data Sources & Setup](#data-sources--setup)
5. [Feature Engineering](#feature-engineering)
6. [Modeling Pipeline](#modeling-pipeline)
7. [Investment Strategies](#investment-strategies)
8. [Alternative Data](#alternative-data)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Installation](#installation)

---

## Project Overview

This project frames WTI crude oil price forecasting as a **binary classification** task: given all information available up to day *t−1*, predict whether the closing price on day *t* is higher or lower than the previous close.

**Five model families** are compared head-to-head:

| Model | Type | Notebook |
|-------|------|----------|
| Elastic Net | Regularized linear | `notebooks/Elastic_Net_Direction_VIP.ipynb` |
| MLP + Elastic Net | Neural net on EN features | `notebooks/MLP_ElasticNet_Direction_VIP.ipynb` |
| XGBoost | Gradient boosting | `notebooks/XGBoost_Direction_VIP.ipynb` |
| LSTM | Recurrent neural net | `notebooks/LSTM_Direction_VIP.ipynb` |
| VAR → threshold | Multivariate time-series | `notebooks/VAR_Direction_VIP.ipynb` |

Every model uses the **same data splits, same feature set, and same preprocessing** — differences in performance are attributable to the model, not the pipeline.

---

## Repository Structure

```
notebooks/
  Elastic_Net_Direction_VIP.ipynb       # Elastic Net classifier
  MLP_ElasticNet_Direction_VIP.ipynb    # MLP on Elastic Net feature extraction
  LSTM_Direction_VIP.ipynb              # LSTM sequence model
  XGBoost_Direction_VIP.ipynb           # XGBoost classifier
  VAR_Direction_VIP.ipynb               # VAR → thresholded binary labels
  Investment_Strategy_VIP.ipynb         # Long-Short, Filtered, Ensemble backtests
  Advanced_Strategy_VIP.ipynb           # Extended strategy analysis
  Sentiment_Pipeline_VIP.ipynb          # FinBERT news sentiment pipeline
  VIP_Abstract_Class.ipynb              # Abstract base class for models
  drafts/                               # Earlier regression prototypes (not graded)

alt_data/
  EIA_Pipeline_VIP.ipynb                # EIA v2 API pull + inventory feature engineering
  XGBoost_EIA_VIP.ipynb                 # XGBoost augmented with EIA inventory features

src/
  data_utils.py                         # Data loading, stationarity transforms, feature builder
  model_classes.py                      # Abstract model class definitions
  script_00_generate_preds.py           # Batch prediction generation across all models
  build_slides.py                       # Presentation build helper

data/
  price/Price.csv                       # WTI daily closing prices (included)
  fred_md_2025_12.csv                   # FRED-MD macro dataset (included)
  figures/                              # Output charts (confusion matrices, equity curves, etc.)

predictions/                            # Saved out-of-sample predictions (val + test CSVs)

report/
  Final_Report.pdf                      # Individual written report

presentations/                          # VIP presentation decks (.pptx)
```

---

## Notebooks & Status

| Notebook | Task | Status |
|----------|------|--------|
| `Elastic_Net_Direction_VIP.ipynb` | Elastic Net classifier | Done |
| `MLP_ElasticNet_Direction_VIP.ipynb` | MLP on EN feature extraction | Done |
| `XGBoost_Direction_VIP.ipynb` | XGBoost classifier | Done |
| `LSTM_Direction_VIP.ipynb` | LSTM sequence model (class imbalance fixed) | Done |
| `VAR_Direction_VIP.ipynb` | VAR forecast → thresholded binary labels | Done |
| `Investment_Strategy_VIP.ipynb` | Long-Short, Confidence-Filtered, Ensemble backtests | Done |
| `Advanced_Strategy_VIP.ipynb` | Extended strategy analysis | Done |
| `Sentiment_Pipeline_VIP.ipynb` | FinBERT sentiment pipeline (integration pending) | Done |
| `alt_data/EIA_Pipeline_VIP.ipynb` | EIA inventory pull + feature engineering | Done |
| `alt_data/XGBoost_EIA_VIP.ipynb` | XGBoost with EIA inventory features | Done |

---

## Data Sources & Setup

### 1. WTI Daily Prices
`data/price/Price.csv` — daily WTI closing prices. Included in the repo.

### 2. FRED-MD Macro Data (31 variables)
`data/fred_md_2025_12.csv` — monthly macro panel. Included in the repo.

The 31 variables used:

```
RPI, W875RX1, CMRMTSPLx, IPFPNSS, USWTRADE, USTRADE, BUSLOANS, CONSPI,
S&P 500, S&P PE ratio, FEDFUNDS, TB3MS, TB6MS, GS1, GS5, GS10, AAA, BAA,
TB3SMFFM, TB6SMFFM, T1YFFM, T5YFFM, T10YFFM, AAAFFM, BAAFFM,
EXSZUSx, EXJPUSx, EXUSUKx, EXCAUSx, PPICMM, UMCSENTx
```

### 3. EIA Petroleum Inventories
Fetched live via the [EIA v2 API](https://www.eia.gov/opendata/) (free API key required). The pipeline notebook handles fetching and local caching automatically.

Key series pulled:

| Series ID | Description |
|-----------|-------------|
| `W_EPC0_SAX_YCUOK_MBBL` | Cushing, OK crude stocks (WTI delivery point) |
| `WCSSTUS1` | Total U.S. commercial crude stocks (excl. SPR) |
| `WGTSTUS1` | Total U.S. gasoline stocks |
| `WDISTUS1` | Total U.S. distillate stocks |

### 4. News Sentiment
Yahoo Finance headlines (21 energy tickers via `yfinance`) scored with [FinBERT](https://huggingface.co/ProsusAI/finbert). See `notebooks/Sentiment_Pipeline_VIP.ipynb`. Note: Yahoo Finance provides only recent headlines, so historical overlap with the full price series is limited.

---

## Feature Engineering

### Price Features
30 lagged daily closing prices: *price_lag01* through *price_lag30* (i.e., *t−1* through *t−30* relative to the prediction date).

### Macro Features
Each of the 31 FRED-MD variables is stationarity-transformed according to its FRED-MD t-code before use:

| T-code | Transformation | Applied to |
|--------|---------------|------------|
| 1 | Level (no change) | `S&P PE ratio`, `UMCSENTx` |
| 2 | First difference | All interest rate and spread variables |
| 5 | Log first difference (growth rate) | Activity, price index, exchange rate variables |

Monthly macro data is forward-filled to the daily trading calendar and shifted forward by one month to respect the **FRED-MD reporting lag** (January values are not available until February 1st).

### EIA Inventory Features (alt data)
Engineered from weekly EIA releases:

| Feature | Description |
|---------|-------------|
| `cushing_chg` | Week-over-week change in Cushing, OK stocks |
| `total_chg` | Week-over-week change in total U.S. crude stocks |
| `cushing_4wma` | 4-week moving average of Cushing stocks |
| `surprise` | `cushing_chg` minus its 4-week rolling mean (consensus deviation) |

EIA data is released every Wednesday (~10:30am ET) for the week ending the previous Friday. It is forward-filled to the daily calendar and then shifted by 1 trading day to prevent same-day look-ahead.

---

## Modeling Pipeline

### Data Splits
- **Train:** up to 2014-12-31
- **Validation:** 2015-01-01 to 2024-12-31
- **Test:** 2025-01-01 onward
- Splits are **time-ordered only** — no shuffling at any stage

### Cross-Validation
5-fold `TimeSeriesSplit` (scikit-learn) within the training set for hyperparameter selection. The scaler is fit on the training fold only — never on the full dataset.

### No-Leakage Rules
- At prediction time *t*, only data from *t−1* and earlier is used
- FRED-MD macro: February predictions use January macro values (one-month lag)
- EIA inventories: Wednesday release data is not used until the following trading day
- `StandardScaler` is fit per fold on training data only

### Target Variable
`y = sign(price_t − price_{t−1})`, with ties mapped to −1 (down). Labels: **1 = up, −1 = down**.

---

## Investment Strategies

Implemented in `notebooks/Investment_Strategy_VIP.ipynb` and `notebooks/Advanced_Strategy_VIP.ipynb`.

| Strategy | Description |
|----------|-------------|
| **Long-Short** | Go long when model predicts up, short when down |
| **Confidence-Filtered** | Trade only when predicted probability exceeds a threshold |
| **Ensemble** | Combine MLP+EN and XGBoost predictions via majority vote |

Backtests include: rolling Sharpe ratio, monthly P&L heatmap, transaction cost sensitivity, and drawdown analysis. All backtests use strictly out-of-sample predictions from the saved `predictions/` CSVs.

---

## Alternative Data

### EIA Petroleum Inventories
The Cushing, OK inventory level is the most market-relevant series because Cushing is the physical delivery point for WTI futures contracts. The `surprise` feature (deviation from recent consensus) is the most predictive: markets price in the expected inventory change; only the unexpected component moves prices.

### FinBERT News Sentiment
21 energy-sector tickers are scraped for recent headlines. Each headline is scored positive/negative/neutral by FinBERT and aggregated to a daily sentiment score. The pipeline design is complete; full historical integration is limited by Yahoo Finance's headline retention window.

---

## Evaluation Metrics

Every model reports the following on strictly out-of-sample predictions:

- Accuracy
- Precision and Recall (per class: up / down)
- F1 score
- ROC-AUC
- Confusion matrix (heatmap)
- At least one additional visualization (equity curve, accuracy over time, feature importance, etc.)

---

## Installation

```bash
pip install numpy pandas scikit-learn xgboost torch transformers yfinance statsmodels matplotlib seaborn requests
```

A GPU is recommended (but not required) for the FinBERT sentiment pipeline. All other notebooks run on CPU.

To regenerate predictions from saved model weights:

```bash
python src/script_00_generate_preds.py
```

---

## Authors

NYU VIP — Energy Commodities Forecasting Group, Spring 2026  
Supervised by Prof. Ed-dib and Idriss Malek
