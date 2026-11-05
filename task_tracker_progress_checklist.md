# 📋 TODO.md — Risk-Sensitive RL Portfolio Allocation

> **Goal:** Build and evaluate a QR-DQN agent for 5-asset portfolio allocation under transaction costs and market regime stress.
> **Rules:** `specs.md` is the single source of truth. Check off subtasks as you complete them. Run and pass all required tests (`T-01` to `T-17`) before moving across phase boundaries.

---

## 🚦 Quick Status Summary

- [ ] **Phase 1:** Data & Feature Pipeline (Steps 01–06)
- [ ] **Phase 2:** Trading Environment & Engine (Steps 07–10)
- [ ] **Phase 3:** Metrics, Benchmarks & Models (Steps 11–19)
- [ ] **Phase 4:** Config Freeze, Backtest & Stress Analysis (Steps 20–23)
- [ ] **Phase 5:** Dashboard & Final Deliverables (Steps 24–29)

---

## 📦 Phase 1: Setup, Data Ingestion & Features

### Step 01 — Spec Review & Team Alignment
- [ ] Read `specs.md` thoroughly with all team members.
- [ ] Agree on key decisions: DL-02 (36-D observation), DL-06 (CVaR action selection), DL-12 (causal rolling features).
- [ ] Confirm and mark all Decision Log entries in `specs.md` as **Accepted**.
- [ ] Confirm defaults for Open Questions (`OQ-01` to `OQ-04`).

### Step 02 — Repository & Environment Setup
- [ ] Create repository directory layout (`config/`, `data/raw/`, `data/processed/`, `artifacts/`, `src/`, `scripts/`, `dashboard/`, `results/`, `reports/`, `tests/`).
- [ ] Create `config/config.yaml` using the skeleton from `specs.md §16`.
- [ ] Create `requirements.txt` with pinned dependencies (`pandas`, `numpy`, `pyarrow`, `pyyaml`, `yfinance`, `gymnasium`, `torch`, `scipy`, `streamlit`, `plotly`, `pytest`).
- [ ] Write `src/utils/seed.py` for reproducible random seeds across Python, NumPy, PyTorch.
- [ ] Create `tests/` folder with stub files for tests `T-01` through `T-17` marked `@pytest.mark.xfail`.
- [ ] **Check:** Run `pip install -r requirements.txt` and `pytest` (must report 17 xfails).

### Step 03 — Data Ingestion & Raw Caching
- [ ] Write `src/data/download.py` using `yfinance.download()`.
- [ ] Tickers: `[NVDA, AAPL, MSFT, GOOGL, AMZN]`.
- [ ] Date window: Start `2012-01-01`, End `2024-01-01` (`interval="1d"`, `auto_adjust=True`).
- [ ] Save output to `data/raw/prices_raw.parquet`.
- [ ] Save metadata to `data/raw/prices_raw.meta.json` (timestamp, `yfinance` version, row count, SHA-256 hash).
- [ ] Add `--refresh` CLI flag (script reads cache by default without making network calls).
- [ ] **Check:** `end="2024-01-01"` is exclusive (last date in dataset is `2023-12-29`).

### Step 04 — Data Cleaning & Quality Checks
- [ ] Write `src/data/clean.py` to align all 5 assets to a single continuous date index.
- [ ] Forward-fill missing prices using `ffill(limit=5)`.
- [ ] Fill missing volume entries with `0`.
- [ ] Save clean dataset to `data/processed/prices_clean.parquet`.
- [ ] Write data quality summary report to `reports/data_quality.md`.
- [ ] Implement unit test `T-01` in `tests/test_data.py`.
- [ ] **Check (`T-01`):** Zero `NaN`/`inf`, strictly increasing unique dates, prices $> 0$, row count between 3000–3030. **No `bfill` or linear interpolation allowed.**

### Step 05 — Feature Extraction (Technical Indicators)
- [ ] Write `src/features/indicators.py` with causal, trailing-only functions:
  - [ ] `ret_1d`: Daily log return $\log(P_t / P_{t-1})$
  - [ ] `sma10_gap`: $P_t / \text{SMA}_{10} - 1$
  - [ ] `sma50_gap`: $P_t / \text{SMA}_{50} - 1$
  - [ ] `rsi14`: Wilder RSI (14-day trailing, `ewm(alpha=1/14, adjust=False)`)
  - [ ] `vol20`: 20-day annualized return volatility
  - [ ] `vol_chg`: 20-day log relative volume change
- [ ] Write `src/features/build_features.py` structuring output as MultiIndex columns `(asset, feature)`.
- [ ] Save raw feature table to `data/processed/features_raw.parquet`.
- [ ] Save warm-up cutoff metadata to `artifacts/feature_meta.json`.
- [ ] Implement causality test `T-02` in `tests/test_features.py`.
- [ ] **Check (`T-02`):** Features at date $t$ computed on truncated data `data[:t+1]` must exactly match features computed on full data.

### Step 06 — Feature Normalization (Z-Score Scaling)
- [ ] Write `src/features/scaler.py` (`fit`, `transform`, `save`, `load`).
- [ ] Compute mean $\mu$ and std $\sigma$ using **training window data ONLY**:
  - [ ] Final Scaler: Fit on 2012–2019 $\rightarrow$ save `artifacts/scaler_2012-2019.json`.
  - [ ] Tuning Scaler: Fit on 2012–2017 $\rightarrow$ save `artifacts/scaler_2012-2017.json`.
- [ ] Apply z-score transform $z = (x - \mu) / \max(\sigma, 10^{-8})$ and clip to $[-5.0, +5.0]$.
- [ ] Export `data/processed/features_z_2012-2019.parquet` and `features_z_2012-2017.parquet`.
- [ ] Write distribution check summary to `reports/feature_shift.md`.
- [ ] Implement unit test `T-03` in `tests/test_scaler.py`.
- [ ] **Check (`T-03`):** Test pipeline loads stored scaler JSON; never calls `fit` on test-period data.

---

## 🕹️ Phase 2: Environment & Trading Mechanics

### Step 07 — Portfolio Engine & Cost Model
- [ ] Write `src/env/portfolio.py` with pure execution logic:
  - [ ] `apply_trades(holdings, target_weights, cost_model)`
  - [ ] `mark_to_market(holdings, asset_returns)`
- [ ] Write `src/env/costs.py` implementing `CostModel(c=0.0010)` (10 bps).
- [ ] Calculate cost against **drifted pre-trade weights** on risky assets only (cash trades are not charged).
- [ ] Deduct fees proportionally from all asset holdings to ensure post-trade weights strictly match targets.
- [ ] Implement unit tests `T-06` and `T-07` in `tests/test_costs.py`.
- [ ] **Check (`T-06` & `T-07`):** 100 consecutive `HOLD` steps incur **$0$ cost**. Moving $10\%$ from NVDA to AAPL at $V_0 = 1.0$ costs exactly $\$0.0002$.

### Step 08 — Action Space & Observation Construction
- [ ] Build 31 discrete action mappings in `artifacts/action_table.json`:
  - [ ] Action 0: `HOLD`
  - [ ] Actions 1–30: Move $10\%$ weight from bucket $i$ to bucket $j$ ($i, j \in [0..5]$, 5 stocks + Cash).
- [ ] Implement action decoder: Move $\min(0.10, w_{\text{pre}}[i])$ from source to target bucket.
- [ ] Construct 36-D observation space: 30 normalized market features (asset-major) + 6 current portfolio weights.
- [ ] Implement unit test `T-04` in `tests/test_env.py`.
- [ ] **Check (`T-04`):** Action table maps bijectively, contains 31 actions, action 0 is `HOLD`, no action has $i = j$.

### Step 09 — Gymnasium Environment Implementation
- [ ] Write `src/env/portfolio_env.py` inheriting from `gymnasium.Env`.
- [ ] Implement `reset(seed, options)` and 5-tuple `step(action)` methods.
- [ ] Support modes: Random 252-day episodes (training) and sequential full-pass (evaluation).
- [ ] Populate diagnostic `info` dictionary (date, portfolio value, turnover, costs paid, weights, drawdown, downside vol).
- [ ] Implement unit tests `T-05` and `T-10` in `tests/test_env.py`.
- [ ] **Check (`T-05` & `T-10`):** Pass `gymnasium.utils.env_checker.check_env`. 10,000 random steps maintain non-negative weights summing to $1.0 \pm 10^{-9}$.

### Step 10 — Risk-Sensitive Reward Function
- [ ] Write `src/env/reward.py` implementing the exact formula:
  $$\text{reward}_t = 100 \times \left( r_{\text{gross}} - \frac{\text{cost}_t}{V_t} - \lambda_1 \cdot \sigma_{\text{down}} - \lambda_2 \cdot \text{DD} \right)$$
- [ ] Calculate $\sigma_{\text{down}}$ as 20-day trailing downside semi-deviation of net daily returns ($\text{MAR} = 0$).
- [ ] Calculate $\text{DD}$ as peak-to-trough drawdown within the current episode.
- [ ] Write diagnostic tracker `src/evaluation/diagnostics.py` (mean cash, % HOLD actions, entropy, turnover).
- [ ] Implement unit tests `T-08` and `T-09` in `tests/test_reward.py`.
- [ ] **Check (`T-08` & `T-09`):** Cost is subtracted exactly once. An `always-HOLD` policy yields identical returns to Buy-and-Hold.

---

## 🧠 Phase 3: Metrics, Benchmarks & Model Training

### Step 11 — Evaluation Metrics Module
- [ ] Write `src/evaluation/metrics.py` providing unified functions for:
  - [ ] Total Return, CAGR (252 days/year)
  - [ ] Annualized Volatility, Sharpe Ratio ($r_f = 0$)
  - [ ] Sortino Ratio, Max Drawdown, Calmar Ratio
  - [ ] Daily CVaR 5% (historical left-tail mean)
  - [ ] Average daily turnover, total transaction fees
  - [ ] 63-day rolling Sharpe & Sortino ratios
- [ ] Implement unit test `T-14` in `tests/test_metrics.py`.
- [ ] **Check (`T-14`):** Output matches hand-calculated benchmark values on a dummy daily return vector.

### Step 12 — Classical Baseline Strategies
- [ ] Write `src/benchmarks/buy_hold.py`: Equal initial weights, zero trades.
- [ ] Write `src/benchmarks/equal_weight.py`: Rebalance to equal weight ($1/N = 0.20$) every 21 trading days.
- [ ] Write `src/benchmarks/mvo.py`: Mean-Variance Optimization (SLSQP, max Sharpe, 252-day window, rebalance every 21 days).
- [ ] Route all benchmarks through the shared `apply_trades()` and `CostModel` functions.
- [ ] Implement unit test `T-13` in `tests/test_benchmarks.py`.
- [ ] **Check (`T-13`):** Identical rebalance schedules in Gym environment and benchmark simulator yield identical equity curves.

### Step 13 — Training Infrastructure & Replay Memory
- [ ] Write `src/agents/replay.py`: Uniform Replay Buffer (`capacity=100,000`).
- [ ] Write `src/agents/train.py`: Unified CLI runner with CSV logging and random seed setting.
- [ ] Write greedy backtester `src/evaluation/backtest.py` ($\varepsilon = 0$).
- [ ] Add check in `backtest.py` preventing execution on out-of-sample test dates unless executed via `run_final_backtest.py`.
- [ ] **Check:** Training logger exports standard metrics and all 4 degeneracy checks to `results/training/`.

### Step 14 — Standard Deep Q-Network (DQN)
- [ ] Write `src/agents/dqn.py`: MLP architecture `[36 -> 128 -> 128 -> 31]`, ReLU activations.
- [ ] Implement Huber Loss ($\kappa = 1.0$), gradient norm clipping at $10.0$, hard target updates every 500 steps.
- [ ] Support linear $\varepsilon$-greedy decay ($1.0 \rightarrow 0.05$ over first $30\%$ of training).
- [ ] Write sanity test on a 2-state toy MDP.
- [ ] **Check:** DQN policy successfully outperforms a random policy on a 60-day training slice without action collapse.

### Step 15 — Hyperparameter Tuning ($\lambda_1, \lambda_2$)
- [ ] Execute grid search over $\lambda_1 \in \{0, 0.05, 0.1, 0.25\}$ and $\lambda_2 \in \{0, 0.002, 0.005, 0.01\}$.
- [ ] Train DQN on 2012–2017 (`tune_train`), evaluate on 2018–2019 (`validation`).
- [ ] Filter out degenerate hyperparameter cells (Mean cash $> 80\%$, HOLD actions $> 95\%$, Entropy $\approx 0$, Daily turnover $> 0.5$).
- [ ] Select best validation Sortino cell, save results to `results/tuning/lambda_grid.csv`.
- [ ] Update selected $\lambda_1, \lambda_2$ values in `config/config.yaml`.
- [ ] **Check:** $\lambda_1, \lambda_2$ are explicitly populated in `config.yaml` and logged in `specs.md §18.1`.

### Step 16 — Tabular Q-Learning Baseline
- [ ] Implement state discretization in `src/agents/tabular_q.py` ($135$ discrete states):
  - [ ] Volatility regime (3 bins, train-set terciles)
  - [ ] Trend regime (3 bins, train-set terciles)
  - [ ] Asset leader (5 assets, max SMA50 gap)
  - [ ] Cash allocation bucket (3 bins: $<5\%$, $5–50\%$, $\ge 50\%$)
- [ ] Save bin cutoffs to `artifacts/tabular_cuts.json`.
- [ ] Train across seeds 0–4 ($\alpha = 0.1$, same action space and $\varepsilon$ schedule).
- [ ] **Check:** All 135 state index calculations are pure, deterministic, and fully unit-tested.

### Step 17 — Quantile Regression DQN (QR-DQN)
- [ ] Write `src/agents/qrdqn.py`: Head output shape `[batch, 31, 20]` ($N = 20$ quantiles).
- [ ] Implement Quantile Huber Loss over all 20 quantiles (tensor shape `[B, N, N]`).
- [ ] Implement CVaR action selection:
  - [ ] **Sort quantile outputs in ascending order** along quantile dimension.
  - [ ] Average lowest $k = \text{ceil}(\alpha \cdot N)$ quantiles ($\alpha = 0.2 \rightarrow k = 4$).
  - [ ] Use CVaR distortion for both acting and TD target greedy choice $a^*$.
- [ ] Implement unit tests `T-11` and `T-12` in `tests/test_qrdqn.py`.
- [ ] **Check (`T-11` & `T-12`):** Learned quantiles match analytic Gaussian inverse CDF on a bandit task. Unsorted quantile tensors are automatically sorted before slicing.

### Step 18 — Model Comparison & Validation Phase
- [ ] Train all 4 target models across 5 seeds (`0–4`) on 2012–2017: `Tabular-Q`, `DQN`, `QR-DQN(α=1.0)`, `QR-DQN(α=0.2)`.
- [ ] Evaluate all runs on 2018–2019 validation set.
- [ ] Determine optimal total training step count per model without touching test data.
- [ ] Export run metrics and update `config/config.yaml`.
- [ ] **Check:** 20 complete training logs created in `results/training/`. All runs pass the degeneracy guard.

### Step 19 — Retraining Full In-Sample Models
- [ ] Retrain all 4 model architectures $\times$ 5 seeds on the entire training set (2012–2019).
- [ ] Use full training scaler `artifacts/scaler_2012-2019.json`.
- [ ] Save 20 model checkpoints to `artifacts/models/{model}_seed{k}.pt`.
- [ ] **Check:** All 20 model checkpoints exist and embed the current `config.yaml` hash.

---

## 🔒 Phase 4: Config Freeze, Backtest & Stress Testing

### Step 20 — 🔒 Configuration Freeze
- [ ] Verify unit tests `T-01` through `T-14` are green.
- [ ] Confirm all 20 model checkpoints exist in `artifacts/models/`.
- [ ] Write and run `scripts/freeze_config.py`.
- [ ] Generate `config/FROZEN` containing SHA-256 hash of `config/config.yaml`.
- [ ] Create git tag `config-frozen`.
- [ ] **Check:** Config freeze completed successfully **before** any test set evaluations are performed.

### Step 21 — Out-of-Sample Test Set Backtest (2020–2023)
- [ ] Write `scripts/run_final_backtest.py`.
- [ ] Implement unit test `T-15` in `tests/test_freeze.py`.
- [ ] Run out-of-sample backtest across all 4 models (5 seeds each) + 3 benchmarks for 2020-01-02 to 2023-12-29.
- [ ] Export primary backtest CSV files to `results/backtests/{strategy}.csv`.
- [ ] Aggregate seed runs into `{model}_mean.csv`.
- [ ] Log entry in Decision Log: "Test set evaluated on [date] under commit [hash]".
- [ ] **Check (`T-15`):** `run_final_backtest.py` strictly refuses to execute if `config/FROZEN` is missing or mismatched.

### Step 22 — Stress Regimes & Slippage Simulation
- [ ] Write `src/evaluation/regimes.py` defining window masks:
  - [ ] `shock_covid`: 2020-02-19 to 2020-03-23
  - [ ] `shock_rates`: 2022-01-03 to 2022-10-12
  - [ ] `normal`: All remaining out-of-sample dates
- [ ] Calculate empirical regime metrics (mean pairwise correlation, volatility ratio vs training baseline).
- [ ] Add dynamic volatility-scaled slippage model to `src/env/costs.py`:
  $$\text{slip\_bps}_{i,t} = s_0 \cdot \text{clip}\left(\frac{\text{vol20}_{i,t}}{\text{median\_train}(\text{vol20}_i)}, 1.0, 5.0\right), \quad s_0 = 2\text{ bps}$$
- [ ] Implement unit test `T-16` in `tests/test_slippage.py`.
- [ ] Re-run backtests under stressed slippage mode $\rightarrow$ export to `results/backtests_slippage/`.
- [ ] Save consolidated summary table to `results/metrics_summary.csv`.
- [ ] **Check (`T-16`):** Median transaction slippage during shock windows is strictly higher than during calm windows.

### Step 23 — Defensive Analysis & Stress Reporting
- [ ] Calculate cash allocation and portfolio volatility in 20-day pre-shock and in-shock windows.
- [ ] Measure maximum in-window drawdown vs Buy-and-Hold and $1/N$.
- [ ] Generate comparative drawdown and position stack plots.
- [ ] Compile stress findings, hypothesis verdicts (H1–H3), and limitations (`L-01` to `L-06`) into `reports/stress_test.md`.
- [ ] **Check:** Every claim and statistic in the stress report links directly to a reproducible script output.

---

## 📊 Phase 5: Dashboard, Audit & Final Deliverables

### Step 24 — Final Data Export & Manifest Generation
- [ ] Write `src/evaluation/export.py` to finalize 15-column CSV layout for all backtests:
  `[date, portfolio_value, daily_return, cost_paid, turnover, w_NVDA, w_AAPL, w_MSFT, w_GOOGL, w_AMZN, w_CASH, regime, drawdown, rolling_sharpe_63, rolling_sortino_63]`
- [ ] Create `results/manifest.json` recording config hash, git commit SHA, execution timestamp, and file list.
- [ ] **Check:** All exported backtest CSVs strictly follow schema contract `C-01`.

### Step 25 — Dashboard Infrastructure & Safe Loader
- [ ] Write `dashboard/loaders.py` using `@st.cache_data`.
- [ ] Validate standard column schema `C-01`/`C-02` during CSV parsing.
- [ ] Write `dashboard/app.py` with sidebar options (Strategy, Regime filter, Slippage toggle).
- [ ] Implement unit test `T-17` in `tests/test_dashboard.py`.
- [ ] **Check (`T-17`):** Dashboard handles missing or malformed CSV files gracefully with an error banner; **never throws an unhandled exception or stack trace.**

### Step 26 — Interactive Dashboard Views
- [ ] Page 1: Cumulative Equity Curves (log-scale toggle, benchmark comparisons).
- [ ] Page 2: Rolling Risk Metrics (63-day Sharpe & Sortino time series).
- [ ] Page 3: Peak-to-Trough Drawdown Analysis.
- [ ] Page 4: Risk Tear Sheet (regime breakdown & slippage toggle).
- [ ] Page 5: Portfolio Weight Allocation Over Time (stacked area plot + seed variance band).
- [ ] **Check:** Dashboard displays all 5 analytical pages using exported CSV data without triggering online model inference or external network calls.

### Step 27 — Dashboard Resilience Verification
- [ ] Test page refreshes and multi-select combinations across options.
- [ ] Test dashboard response when an expected CSV file is temporarily renamed or removed.
- [ ] **Check:** App displays user-friendly error messages and maintains stability under all interaction tests.

### Step 28 — Reproducibility Audit & Documentation
- [ ] Write `README.md` containing setup commands, `make` pipeline targets, and guide to `specs.md`/`TODO.md`.
- [ ] Perform full reproduction run from a clean repository clone (`make all`).
- [ ] Verify all automated unit tests pass (`pytest`).
- [ ] **Check:** Complete pipeline runs end-to-end from scratch and reproduces identical metrics in `results/metrics_summary.csv`.

### Step 29 — Final Report & Project Sign-Off
- [ ] Compile final results, methodology, hypotheses conclusions, Decision Log, and limitations into `reports/final_report.md`.
- [ ] Audit every numeric claim in `reports/final_report.md` against `results/manifest.json`.
- [ ] Update project status: **Project Complete**.
- [ ] **Check:** Zero untraceable figures or statistics in the final report.

---

## 🚫 Blockers & Issues Log

| Date | Step | Issue / Blocker Description | Owner | Status | Resolution |
|---|---|---|---|---|---|
| — | — | *No active blockers logged.* | — | — | — |

---

## 📝 Session & Audit Log

| Date | Developer | Task / Changes Completed | Commit Hash | Next Required Step |
|---|---|---|---|---|
| 2026-11-06 | Team | Initialized project structure, `specs.md` v1.0, and `TODO.md` checklist. | `initial` | Step 01 |