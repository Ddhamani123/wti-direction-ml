# WTI Crude Oil Direction Classification

**NYU VIP — ML in Energy Commodities (Spring 2026)**

Binary next-day price direction prediction (↑ / ↓) for WTI crude oil using four model families, two investment strategies, and two alternative data sources.

---

## Project Overview

This project frames WTI crude oil price forecasting as a **classification** problem — predicting whether tomorrow's price is up or down — rather than a regression task. Models are evaluated strictly out-of-sample using 5-fold time-series cross-validation.

**Models:** Elastic Net · MLP+Elastic Net · XGBoost · LSTM · VAR (thresholded)  
**Investment Strategies:** Long-Short · Confidence-Filtered · Ensemble  
**Alternative Data:** EIA Petroleum Inventories · FinBERT News Sentiment  

---

## Repository Structure

```
notebooks/
  Elastic_Net_Direction_VIP.ipynb       # Elastic Net classifier
  MLP_ElasticNet_Direction_VIP.ipynb    # MLP on top of EN feature extraction
  LSTM_Direction_VIP.ipynb              # LSTM sequence model
  XGBoost_Direction_VIP.ipynb           # XGBoost classifier
  VAR_Direction_VIP.ipynb               # VAR → thresholded binary labels
  Investment_Strategy_VIP.ipynb         # Backtests: Long-Short, Filtered, Ensemble
  Sentiment_Pipeline_VIP.ipynb          # FinBERT sentiment feature pipeline
  Advanced_Strategy_VIP.ipynb           # Extended strategy analysis
  VIP_Abstract_Class.ipynb              # Abstract base class for models
  drafts/                               # Earlier price-prediction (regression) prototypes

alt_data/
  EIA_Pipeline_VIP.ipynb                # EIA v2 API pull + feature engineering
  XGBoost_EIA_VIP.ipynb                 # XGBoost augmented with EIA inventory features

src/
  data_utils.py                         # Data loading, stationarity transforms, feature builder
  model_classes.py                      # Abstract model class definitions
  script_00_generate_preds.py           # Batch prediction generation script
  build_slides.py                       # Presentation build helper

data/
  price/Price.csv                       # WTI daily closing prices
  figures/                              # Output charts (confusion matrices, equity curves, etc.)

predictions/                            # Out-of-sample model predictions (val + test)

report/
  main.tex                              # LaTeX individual report source

presentations/                          # VIP presentation decks (.pptx)
```

---

## Data Sources

### 1. WTI Daily Prices
`data/price/Price.csv` — included in repo.

### 2. FRED-MD Macro Vintages (31 variables)
Historical monthly macro data. **Not included in repo** (too large). Download from:
- **1999–2014:** [FRED-MD Historical Vintages](https://research.stlouisfed.org/wp/more/2015-012)
- **2015–2024:** [FRED-MD Current Vintages](https://fred.stlouisfed.org/releases/download?rid=2081&vintage=2024-12&transform=lin)

Place downloaded files in:
```
Historical_FRED-MD/Historical FRED-MD Vintages Final/   ← 1999–2014 monthly CSVs
Historical-vintages-of-FRED-MD-2015-01-to-2024-12/      ← 2015–2024 monthly CSVs
```

The 31 FRED-MD features used: `RPI, W875RX1, CMRMTSPLx, IPFPNSS, USWTRADE, USTRADE, BUSLOANS, CONSPI, S&P 500, S&P PE ratio, FEDFUNDS, TB3MS, TB6MS, GS1, GS5, GS10, AAA, BAA, TB3SMFFM, TB6SMFFM, T1YFFM, T5YFFM, T10YFFM, AAAFFM, BAAFFM, EXSZUSx, EXJPUSx, EXUSUKx, EXCAUSx, PPICMM, UMCSENTx`

### 3. EIA Petroleum Inventories
Pulled via the [EIA v2 API](https://www.eia.gov/opendata/) (free key required). The pipeline notebook (`alt_data/EIA_Pipeline_VIP.ipynb`) handles fetching and caching automatically.

### 4. News Sentiment
Yahoo Finance headlines scored with [FinBERT](https://huggingface.co/ProsusAI/finbert) via `yfinance`. See `notebooks/Sentiment_Pipeline_VIP.ipynb`.

---

## Key Design Rules

- **No data leakage:** predictions at time *t* use only data from *t−1* and earlier
- **FRED-MD one-month lag:** macro data has a reporting delay; February predictions use January's values
- **Time-based splits only:** 5-fold `TimeSeriesSplit` — never shuffle
- **Scaler fit on training data per fold** — never on the full dataset
- **30-day price lookback:** features include closing prices *t−1* through *t−30*

---

## Metrics (all models)

Accuracy · Precision · Recall (per class) · F1 · ROC-AUC · Confusion Matrix

---

## Setup

```bash
pip install numpy pandas scikit-learn xgboost torch transformers yfinance statsmodels matplotlib seaborn
```

For the FinBERT sentiment pipeline, a GPU is recommended but not required.

---

## Authors

NYU VIP — Energy Commodities Forecasting Group, Spring 2026  
Supervised by Prof. Ed-dib and Idriss Malek
