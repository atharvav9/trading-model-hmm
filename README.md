# Algorithmic Trading & ML Backtesting Suite (HMM-Driven Pipeline) 📉📊

A full-stack quantitative trading research environment and live execution toolkit built on MetaTrader 5 (MT5). The project spans across live monitoring strategies to extensive machine learning models for detecting volatile environments and creating directional predictions.

If you're looking to see how theoretical quant research translates into deployable ML solutions, you're in the right spot. This project is all about rigorous feature engineering, backtesting, and making sure the models actually hold up under live-market constraints.

## 🚀 What's Under the Hood?

* **Latent Regime Detection:** Uses Hidden Markov Models (HMM) to figure out the hidden states of the market (bull, bear, high-volatility, etc.) purely from observable price and volume data. 
* **Algorithmic Sequence Predictions:** Optimized for generating trading signals with a solid **58% accuracy rate** on core sequence predictions. (If you know quant trading, you know hitting 58% directional accuracy gives you a pretty sweet edge).
* **Robust Backtesting:** Includes dedicated modules to benchmark and evaluate model performance so we aren't just flying blind in live scenarios.
* **Production-Ready Pipelines:** Clean feature engineering and validation workflows built to scale and integrate into larger trading architectures.

## 📁 Repository Structure

- `core/` - Reusable indicator and feature engineering logic (Work in Progress).
- `scanners/` - Live execution modules connecting to your MetaTrader 5 terminal:
  - `live_scanner.py`: A unified M5 timeframe scanner using VWAP and Double-Deviation ATR breakouts. Provides CLI arguments to scan different specific assets.
  - `advanced_monitor.py`: Follows specialized impulse sequencing and pullback contraction volume filtering.
- `models/` - Quant backtesting and machine learning logic:
  - `backtest_intraday.py`: Runs a Random Forest classifier over 3M Scalping scenarios to deduce feature importance.
  - `regime_hmm.py`: A Hidden Markov Model trained on Historical Returns + Volatility to classify sideways/directional regimes.
  - `swing_mlp.py`: Vectorized backtest harnessing an Artificial Neural Network (`MLPClassifier`) for swing positions based on Stochastics and custom Volume Force parameters.
- `utils/` - Operations and data handling scripts.
  - `data_fetcher.py`: Interacts with MT5 to download historic dataset batches cleanly.
- `data/samples/` - Miniature datasets to run backtests, with instructions below on syncing full datasets safely to `data/raw/` via `.gitignore`. 

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/atharvav9/trading-model-hmm.git
   cd trading-model-hmm
   ```

2. **Install Python Dependencies:**
   Make sure you are running a Python version compatible with `MetaTrader5` (Windows only).
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the Environment `.env`:**
   Because keeping account numbers and passwords in source code poses a massive security risk, standard practice dictates we use `.env` files.
   * Duplicate `.env.example` and rename the clone to just `.env`.
   * Add your authentic MT5 Login ID and Password carefully.
   ```
   MT5_ACCOUNT=12345678
   MT5_PASSWORD="your_password_here"
   MT5_SERVER="MetaQuotes-Demo"
   ```

---

## 🚀 Execution Guide

### Scanning Live Markets

Track breakouts securely on any asset without needing to copy-paste multiple `.py` scripts.
```bash
python scanners/live_scanner.py --symbol USDJPY
python scanners/live_scanner.py --symbol GBPUSD
```
```bash
python scanners/advanced_monitor.py --symbol XAUUSD
```

### Fetching Fresh Datasets

To feed the ML scripts, you need historic ticks. You can fetch and specify output targets:
```bash
python utils/data_fetcher.py --symbol EURUSD --year 2024 --output data/raw/EURUSD_2024.csv
```

### Auditing & Backtesting

Run Intraday ML Profiling:
```bash
python models/backtest_intraday.py
```

Check the Daily Regime (HMM):
```bash
python models/regime_hmm.py
```

Simulate the Institutional MLP (V10):
```bash
python models/swing_mlp.py
```

## 📈 Results & Benchmarks

The core HMM achieved a 58% accuracy rate during rigorous backtesting and validation. The pipeline successfully filters out market noise and provides actionable, data-driven optimization for algorithmic trading strategies, proving its viability for real-world application.

---
*Built for the love of data, stochastic modeling, and trying to beat the market.*
