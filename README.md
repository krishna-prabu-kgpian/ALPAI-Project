# Q-Learning for Quant Finance: Risk-Sensitive Portfolio Allocation & Execution via QR-DQN

**Term Project Proposal — Group 3**

## Team Members

- **Vangala Akshay Reddy** — 23CE3FP11
- **Kusuma Akshaya** — 23EE10032
- **Krishna Prabu** — 23CS30028
- **Srishti Somya** — 23CS30055

## 1. Aim

This project evaluates **value-based Deep Reinforcement Learning (DRL)** for dynamic portfolio allocation and trade execution under market friction, incorporating **10 basis points (bps) transaction costs**.

We compare three reinforcement learning approaches:

- **Tabular Q-Learning**
- **Deep Q-Network (DQN)**
- **Quantile Regression Deep Q-Network (QR-DQN)**

To limit downside risk, step rewards penalize realized volatility and drawdowns.

The primary objective is to investigate whether **distributional Q-learning (QR-DQN)** can outperform traditional portfolio management strategies such as:

- Buy-and-Hold
- Equal-Weight Portfolio (1/N)
- Mean-Variance Optimization (MVO)

The models will be evaluated during historical market regime shifts and liquidity shocks.

## 2. Data Sources

### Asset Universe & Features

Historical financial data will be collected using **yfinance**, covering S&P 500 stocks and/or Sector ETFs.

The state representation will include:

- **Close Price**
- **10-day Simple Moving Average (SMA)**
- **50-day Simple Moving Average (SMA)**
- **14-day Relative Strength Index (RSI)**
- **Portfolio Weights**
- **Market Volatility**

### Data Splits & Stress Testing

The dataset will be divided into:

| Dataset | Period | Purpose |
|---|---|---|
| In-Sample | 2012–2019 | Model training |
| Out-of-Sample | 2020–2023 | Evaluation and stress testing |

The out-of-sample evaluation will specifically cover:

- **COVID-19 market shock (2020)**
- **Interest rate-hike shocks (2022–2023)**

These periods will help evaluate model robustness under changing market conditions.

## 3. Planned Experiments

### 3.1 Model Training

Train and compare the following agents in a custom **Gymnasium** environment:

1. Tabular Q-Learning
2. Deep Q-Network (DQN)
3. Quantile Regression Deep Q-Network (QR-DQN)

### 3.2 Reward Function Tuning

Design and iteratively refine risk-sensitive reward functions that balance:

- Expected portfolio returns
- Downside volatility
- Drawdown penalties
- Transaction costs and execution slippage

The goal is to optimize risk-adjusted portfolio performance rather than solely maximizing raw returns.

### 3.3 Curriculum Learning

Implement curriculum learning by progressively exposing agents to different market conditions:

1. Begin training with low-volatility synthetic market data.
2. Gradually introduce historical crash data.
3. Evaluate policy stability under high-volatility conditions.

This approach aims to improve the robustness and stability of the learned policies.

### 3.4 Benchmarking

Compare the trained RL agents against traditional portfolio strategies:

- **Buy-and-Hold**
- **Equal-Weight Portfolio (1/N)**
- **Markowitz Mean-Variance Optimization (MVO)**

All strategies will be evaluated over identical testing periods to ensure fair comparisons.

### 3.5 Performance Evaluation

Evaluate model performance using the following metrics:

| Metric | Description |
|---|---|
| **Sharpe Ratio** | Measures risk-adjusted returns relative to total volatility |
| **Sortino Ratio** | Measures risk-adjusted returns considering downside volatility |
| **Maximum Drawdown (MDD)** | Measures the largest peak-to-trough portfolio decline |
| **Execution Slippage** | Measures the difference between expected and actual execution prices |

Additional visual analyses will be conducted to understand how portfolio performance and agent behavior change across market regimes.

## 4. Expected Outcomes

The expected deliverables of this project include:

### Risk-Sensitive QR-DQN Agent

A trained **Quantile Regression DQN (QR-DQN)** agent capable of dynamic portfolio allocation while accounting for downside risk and transaction costs.

An optional **LSTM-based feature extractor** may be incorporated to capture temporal dependencies in financial time-series data.

### Automated Risk Tear Sheet

An automated report summarizing portfolio performance, including:

- Risk-adjusted returns
- Drawdown analysis
- Portfolio volatility
- Execution slippage
- Comparative benchmark performance

### Interactive Streamlit Dashboard

An interactive **Streamlit dashboard** for:

- Portfolio performance visualization
- Risk analysis
- Strategy comparisons
- Market stress testing
- Real-time portfolio advisory

## 5. References

1. V. Mnih et al., *Human-level control through deep reinforcement learning*, Nature, 518(7540), 2015.

2. W. Dabney et al., *Distributional Reinforcement Learning with Quantile Regression*, AAAI, 2018.
