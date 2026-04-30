# Cointegrated Pairs Trading

A market-neutral equity pairs trading strategy backtested on S&P 500 constituents (2017–2024), with two signal methods: a classic z-score spread strategy and a novel hybrid RSI+z-score indicator.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

## Overview

The strategy screens S&P 500 stocks by fundamentals, ranks all candidate pairs across seven statistical measures (SSD, price ratio, correlation, covariance, and magnitude-squared coherence), and selects the top 20 cointegrated pairs via Engle-Granger testing. Two trading methods are then backtested over the selected pairs:

- **Z-Score Strategy** — opens positions when the rolling z-score of the price ratio crosses a threshold, closing when it mean-reverts.
- **Indicator Strategy** — combines a sigmoid-normalized z-score with the RSI spread between the two assets into a composite indicator on [0, 100], triggering entries and exits at configurable bounds.

Both methods target near-zero market beta by entering delta-neutral long/short positions on each pair simultaneously. The strategy is designed to outperform during market downturns and exhibits significantly lower drawdowns than passive index investing.

<img width="998" alt="Portfolio vs S&P 500" src="https://github.com/aryamann04/cointegrated-pairs-trading/assets/140534650/294130a0-a2dd-4b72-b25c-6790b42a5ac1">

## Results (2017–2024, 20 equity pairs, unleveraged)

| Strategy  | CAGR  | Sharpe | Beta    | VaR (95%) | CVaR (95%) |
|-----------|-------|--------|---------|-----------|------------|
| Z-Score   | 7.63% | 0.490  | 0.00324 | −0.28%    | −0.65%     |
| Indicator | 8.21% | 0.759  | 0.00306 | −0.27%    | −0.61%     |

Z-Score: z-threshold 1.1, MA window 175 days. Indicator: bounds [40, 52], MA window 150 days.

## Features

- **Fundamental screening** — filters S&P 500 universe by market cap, trailing EPS, and P/E ratio via yfinance
- **7-criteria pair ranking** — composite ranking across SSD, price ratio proximity, log-price correlation, return correlation, log-price covariance, return covariance, and magnitude-squared coherence (Brunetti & DeLuca 2023)
- **Engle-Granger cointegration testing** — selects pairs with p-value < 0.03; final 20 chosen by ratio variance
- **Z-score signal generation** — rolling mean/std of price ratio with configurable window and threshold
- **Hybrid RSI indicator** — `Indicator = 100 × (0.5 × sigmoid(z) + 0.5 × sigmoid((RSI₁ − RSI₂) / 10))`, signals on configurable lower/upper bounds
- **Delta-neutral P&L** — 0.1% transaction fee applied on entry; optional leverage scaling
- **Performance metrics** — Sharpe ratio, CAGR, annualized variance, VaR (95%), CVaR (95%), win rate, market beta
- **Parameter optimization** — grid search over z-threshold × MA window (z-score) or lower/upper bounds (indicator), optimizable for Sharpe or CAGR
- **Commodity pairs support** — pre-selected commodity ETF pairs (PPLT/PALL, WEAT/UNL, etc.) as an alternative universe

## Project Structure

```
cointegrated-pairs-trading/
├── Backtesting.py          # Entry point — runs backtest, plots results, exposes optimizer
├── BuySellSignals.py       # Data loading, z-score signal generation, P&L calculation
├── Indicator.py            # RSI calculation and hybrid indicator signal generation
├── PerformanceMetrics.py   # Sharpe, CAGR, VaR, CVaR, beta, win rate
├── InitialScreening.py     # S&P 500 scraper and fundamental screener
├── PreSelectionTests.py    # 7-measure pair ranking (run once to select pairs)
├── CointegrationTests.py   # Engle-Granger cointegration test and variance ranking
└── requirements.txt
```

## Setup

**Prerequisites:** Python 3.9+

```bash
pip install -r requirements.txt
```

## Usage

**Run the full backtest:**

```bash
python Backtesting.py
```

Loads the 20 pre-selected cointegrated equity pairs, runs both strategies with default parameters, prints performance metrics, and plots cumulative returns vs. the S&P 500.

**Default parameters:**

| Parameter | Value |
|---|---|
| Lookback window (`ma`) | 50 days |
| Z-score threshold | 1.5 |
| Indicator lower bound | 45 |
| Indicator upper bound | 55 |
| Transaction fee | 0.1% |

**Parameter optimization** (uncomment in `Backtesting.py`):

```bash
# Grid search over indicator bounds or z-score/MA window, optimizing Sharpe or CAGR
# Uncomment the relevant line at the bottom of Backtesting.py, then:
python Backtesting.py
```

**Re-running the full pair selection pipeline** (one-time research step):

```python
from InitialScreening import screen
from PreSelectionTests import pairs_measures, rank_pairs
from CointegrationTests import coint_test, rank_var

tickers = screen(mkt_cap=10e9, eps=2.0, pe_low=10, pe_high=40)
measures = pairs_measures(tickers)
ranked = rank_pairs(measures)
cointegrated = coint_test(ranked)
final_pairs = rank_var(cointegrated)
print(final_pairs)
```

**With leverage:**

```python
from Backtesting import run_strategy, generate_dataframes, ranked_df

df_list = generate_dataframes(ranked_df)
portfolio = run_strategy(
    50, df_list,
    use_indicator_strategy=True, lower_bound=45, upper_bound=55,
    use_leverage=True, leverage_ratio=2.0
)
```
