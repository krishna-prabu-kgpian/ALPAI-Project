# TODO.md — Modular, Interrupt-Friendly Task Plan

> **Project:** Two-Stage Risk-Sensitive RL Portfolio Allocator (`specs.md` v1.1)  
> **Rule of Thumb:** Work on **one sub-task at a time**. Tick the box as soon as it passes verification. If interrupted, leave a note in the Session Log indicating the exact sub-task ID.

---

## 1. Execution Overview & Split

### Parallel Work Lanes (After Step 02)

| Lane | Focus | Steps | Prerequisites |
|---|---|---|---|
| **Lane A** | Data → Features → Regime Model | 03 → 04 → 05 → 06 → 07 | Step 02 |
| **Lane B** | Portfolio Engine → Env → Curriculum | 08 → 09 → 10 → 11 → 12 | Step 08 needs 04+05; Step 09 needs 06+07+08 |
| **Lane C** | Metrics → Benchmarks | 13 → 14 | Step 13 needs 04; Step 14 needs 08 |

*Note: Steps 15 onward are sequential and require Lanes A, B, and C to be completed.*

### Time-Box Cut Hierarchy (specs §12 / TU-05)
If running low on time, make cuts in this strict order and log each in the Decision Log:
1. Skip $\lambda$ grid search (Step 17) $\rightarrow$ use default $(0.05, 0.002)$ labeled "untuned default".
2. Validation comparison (Step 20) $\rightarrow$ 1 seed instead of 2.
3. Final seeds $\rightarrow$ reduced from 3 to 2.
4. Training steps $\rightarrow$ reduced from 150,000 to 100,000.
5. Drop control `A2` (no-curriculum run).
6. Drop control `A1` (one-stage run, no regime). Mark hypotheses H4/H5 as "untested".

---

## 2. Status Board

*Status options: `TODO` · `IN_PROGRESS` · `DONE` · `BLOCKED`*

| Step ID | Step Title | Lane | Depends On | Status | Owner |
|---|---|---|---|---|---|
| **01** | Spec Review & Decision Log Sign-off | Core | 00 | `DONE` | All |
| **02** | Repo Structure, Environment & Config | Core | 01 | `DONE` | |
| **03** | Price Ingestion & Caching | Lane A | 02 | `DONE` | |
| **04** | Price Cleaning & Data Quality Proof | Lane A | 03 | `DONE` | |
| **05** | Stock & Market Feature Engineering | Lane A | 04 | `DONE` | |
| **06** | Fit-Window Feature Normalization (z-score) | Lane A | 05 | `DONE` | |
| **07** | Stage 1 Market-Regime Model (GMM) | Lane A | 06 | `TODO` | |
| **08** | Portfolio Engine & Cost Model | Lane B | 04, 05 | `TODO` | |
| **09** | Action Space & Observation Construction | Lane B | 06, 07, 08 | `TODO` | |
| **10** | Gymnasium Environment Construction | Lane B | 09 | `TODO` | |
| **11** | Risk-Sensitive Reward Function | Lane B | 10 | `TODO` | |
| **12** | Curriculum Sampler & Split Integrity | Lane B | 10 | `TODO` | |
| **13** | Metrics Calculation Engine | Lane C | 04 | `TODO` | |
| **14** | Baseline Benchmarks (B&H, 1/N, MVO) | Lane C | 08, 13 | `TODO` | |
| **15** | Training Infrastructure & Time Budget | Core | 11, 12, 13 | `TODO` | |
| **16** | Standard Deep Q-Network (DQN) | Core | 15 | `TODO` | |
| **17** | Reward Penalty Grid Search ($\lambda_1, \lambda_2$) | Core | 16 | `TODO` | |
| **18** | Compressed-State Tabular Q-Learner | Core | 15, 17 | `TODO` | |
| **19** | Distributional Quantile RL (QR-DQN) | Core | 16, 17 | `TODO` | |
| **20** | Validation Scoring & Headline Selection | Core | 18, 19 | `TODO` | |
| **21** | Full Curriculum Training Run | Core | 20 | `TODO` | |
| **22** | Configuration & Codebase Freeze | Core | 14, 21 | `TODO` | |
| **23** | Blind Backtest Execution | Core | 22 | `TODO` | |
| **24** | Post-Freeze Diagnostic & Slippage Analysis | Core | 23 | `TODO` | |
| **25** | Success Criteria & Hypothesis Verdicts | Core | 24 | `TODO` | |
| **26** | Result Manifest & Data Export | Core | 24 | `TODO` | |
| **27** | Streamlit Core Architecture & Data Loaders | Core | 26 | `TODO` | |
| **28** | Streamlit Interactive Views | Core | 27 | `TODO` | |
| **29** | Streamlit Error-Handling Audit | Core | 28 | `TODO` | |
| **30** | Reproduction Test Suite & Execution Scripts | Core | 29 | `TODO` | |
| **31** | Final Audit & Project Report | Core | 25, 30 | `TODO` | |

---

## 3. Step Details

---

### Step 01 — Spec Review & Decision Log Sign-off
* **Prerequisites:** Step 00 (`specs.md` v1.1)
* **Goal:** Align all team members on rules, split dates, and pre-registered criteria prior to writing code.

* **Sub-tasks:**
  - [x] **1.1** Read `specs.md` completely (§0, §1, §2, §5, §7, §10, §21).
  - [x] **1.2** Review Decisions `DL-16` through `DL-26` with team.
  - [x] **1.3** Confirm Open Questions `OQ-01` through `OQ-09` defaults.
  - [x] **1.4** Update `specs.md` §21.1 Decision Log rows from *Proposed* to *Accepted*.

* **Artifacts Created:** Updated `specs.md`
* **Verification Criteria:** No Decision Log row remains in *Proposed* state.

---

### Step 02 — Repo Structure, Environment & Config
* **Prerequisites:** Step 01
* **Goal:** Instantiate project directory, configuration definitions, and test execution scaffold.

* **Sub-tasks:**
  - [x] **2.1** Create directory structure per `specs.md` §20 (`data/raw`, `data/processed`, `artifacts/models`, `src/`, `tests/`, `reports/`, `results/`).
  - [x] **2.2** Copy configuration skeleton from `specs.md` §19 into `config/config.yaml`.
  - [x] **2.3** Populate `requirements.txt` with pinned dependencies (`pandas`, `numpy`, `pyarrow`, `pyyaml`, `yfinance`, `gymnasium`, `torch`, `scikit-learn`, `joblib`, `scipy`, `streamlit`, `plotly`, `pytest`).
  - [x] **2.4** Implement standard seed utility `src/utils/seed.py` for Python `random`, `numpy`, and `torch`.
  - [x] **2.5** Create placeholder pytest files `tests/test_*.py` for tests `T-01` through `T-20` marked `@pytest.mark.xfail`.
  - [x] **2.6** Create `Makefile` with basic execution targets (`clean`, `test`).

* **Artifacts Created:** Directory tree, `config/config.yaml`, `requirements.txt`, `src/utils/seed.py`, `tests/test_*.py`, `Makefile`
* **Verification Criteria:** Executing `pytest` in a fresh virtualenv reports 20 `xfail` items without errors.

---

### Step 03 — Price Ingestion & Caching
* **Prerequisites:** Step 02 | **Lane:** A
* **Goal:** Download asset OHLCV data from Yahoo Finance and cache locally for reproducible offline access.

* **Sub-tasks:**
  - [x] **3.1** Write `yfinance` probe script to confirm returned column formats for multi-ticker queries.
  - [x] **3.2** Create `src/data/download.py` using parameters from `config.yaml` (`tickers`, `start="2012-01-01"`, `end="2026-05-01"`, `auto_adjust=True`).
  - [x] **3.3** Implement `--refresh` flag logic: bypass network and load disk cache if flag is absent.
  - [x] **3.4** Write raw dataset metadata to `data/raw/prices_raw.meta.json` containing SHA-256 hash, row count, timestamp, and library versions.

* **Artifacts Created:** `src/data/download.py`, `data/raw/prices_raw.parquet`, `data/raw/prices_raw.meta.json`
* **Verification Criteria:** Raw file exists; re-running without `--refresh` makes zero network requests; final date index entry is `2026-04-30`.

---

### Step 04 — Price Cleaning & Data Quality Proof
* **Prerequisites:** Step 03 | **Lane:** A
* **Goal:** Reindex asset price data onto a unified calendar index without look-ahead bias.

* **Sub-tasks:**
  - [x] **4.1** Write `src/data/clean.py` to align all 5 assets on a shared calendar index.
  - [x] **4.2** Apply `ffill(limit=5)` to price columns; fill missing volume entries with `0`. (Strictly prohibition: no `bfill` or spline interpolation).
  - [x] **4.3** Save cleaned output to `data/processed/prices_clean.parquet`.
  - [x] **4.4** Write automated data sanity test `tests/test_data.py` (`T-01`): verify zero `NaN`/`inf`, strictly ascending dates, all prices $> 0$.
  - [x] **4.5** Create `reports/data_quality.md` documenting missing cell fill counts and top 10 single-day return movements per asset.

* **Artifacts Created:** `src/data/clean.py`, `data/processed/prices_clean.parquet`, `reports/data_quality.md`, `tests/test_data.py`
* **Verification Criteria:** Test `T-01` passes; total filled cells $< 0.1\%$; single-day movements $> 25\%$ explained in report.

---

### Step 05 — Stock & Market Feature Engineering
* **Prerequisites:** Step 04 | **Lane:** A
* **Goal:** Compute stationary per-stock indicators and macro market features on continuous time-series data.

* **Sub-tasks:**
  - [x] **5.1** Implement `src/features/indicators.py` calculating `ret_1d`, `sma10_gap`, `sma50_gap`, `vol20`, and `vol_chg` on trailing windows.
  - [x] **5.2** Implement Wilder RSI 14 in `indicators.py` using `ewm(alpha=1/14, adjust=False)`.
  - [x] **5.3** Implement `src/features/market_features.py` computing basket features `m_vol20`, `m_corr20`, and trailing 252-day `m_dd`.
  - [x] **5.4** Construct pipeline script `src/features/build_features.py` generating `data/processed/features_raw.parquet` and `data/processed/market_features_raw.parquet`.
  - [x] **5.5** Drop leading warm-up rows containing `NaN` and output initial valid index date to `artifacts/feature_meta.json`.
  - [x] **5.6** Implement test `T-02` in `tests/test_features.py`: verify trailing window feature equality between truncated sub-series `data[:t+1]` and full dataset across $\ge 50$ random cut dates.

* **Artifacts Created:** `src/features/indicators.py`, `src/features/market_features.py`, `src/features/build_features.py`, `data/processed/features_raw.parquet`, `data/processed/market_features_raw.parquet`, `artifacts/feature_meta.json`
* **Verification Criteria:** Test `T-02` passes; output contains zero NaN values after warm-up drop; first valid row date aligns near mid-March 2012.

---

### Step 06 — Fit-Window Feature Normalization (z-score)
* **Prerequisites:** Step 05 | **Lane:** A
* **Goal:** Normalize features to zero mean and unit variance using statistics derived strictly from training windows.

* **Sub-tasks:**
  - [x] **6.1** Create standard scaler module `src/features/scaler.py` with `fit`, `transform`, `save`, and `load` methods, standard deviation floor $10^{-8}$, and fixed clipping bound $[-5.0, +5.0]$.
  - [x] **6.2** Fit scaler on **2012–2023** window and write parameters to `artifacts/scaler_2012-2023.json`.
  - [x] **6.3** Fit scaler on **2012–2021** window and write parameters to `artifacts/scaler_2012-2021.json`.
  - [x] **6.4** Transform features to generate `data/processed/features_z_2012-2023.parquet` and `data/processed/features_z_2012-2021.parquet`.
  - [x] **6.5** Write test `T-03` in `tests/test_scaler.py`: confirm transformation parameters are strictly locked to fit-window bounds.

* **Artifacts Created:** `src/features/scaler.py`, scaler JSON artifacts, normalized feature parquets, `tests/test_scaler.py`
* **Verification Criteria:** Test `T-03` passes; normalized features over fit window exhibit mean $\approx 0$ and standard deviation $\approx 1$.

---

### Step 07 — Stage 1 Market-Regime Model (GMM)
* **Prerequisites:** Step 06 | **Lane:** A
* **Goal:** Train a Gaussian Mixture Model to output daily market regime probabilities without look-ahead leakage.

* **Sub-tasks:**
  - [ ] **7.1** Implement `src/regime/regime_model.py`: fit 3-component diagonal GMM (`n_init=5`, `random_state=0`) on z-scored market features.
  - [ ] **7.2** Implement component sorting by ascending mean `m_vol20` ($0=\text{Calm}$, $1=\text{Choppy}$, $2=\text{Crisis}$).
  - [ ] **7.3** Implement causal EMA smoothing ($\text{span}=5$) across per-day posterior probabilities (`predict_proba`).
  - [ ] **7.4** Write `src/regime/build_regime.py` fitting models for both final (`2012-2023`) and tuning (`2012-2021`) windows, saving `artifacts/regime_{fit_window}.joblib` and `data/processed/regime_probs_{fit_window}.parquet`.
  - [ ] **7.5** Write test `T-18` in `tests/test_regime.py`: verify regime ordering, probability row sums equal $1.0$, and strict posterior causality.
  - [ ] **7.6** Generate `reports/regime_report.md` summarizing regime proportions and feature means across fit/validation windows.

* **Artifacts Created:** `src/regime/regime_model.py`, `src/regime/build_regime.py`, `artifacts/regime_*.joblib`, `data/processed/regime_probs_*.parquet`, `reports/regime_report.md`, `tests/test_regime.py`
* **Verification Criteria:** Test `T-18` passes; Crisis regime exhibits highest mean volatility and drawdown; report contains no post-2023 dates.

---

### Step 08 — Portfolio Engine & Cost Model
* **Prerequisites:** Steps 04, 05 | **Lane:** B
* **Goal:** Build the deterministic portfolio transaction, valuation, and transaction fee engine.

* **Sub-tasks:**
  - [ ] **8.1** Create pure valuation functions `src/env/portfolio.py`: `apply_trades(holdings, target_weights, cost_model)` and `mark_to_market(holdings, returns)`.
  - [ ] **8.2** Create transaction fee model `src/env/costs.py` enforcing $10\text{ bps}$ ($c=0.0010$) on traded notional of risky assets against drifted pre-trade weights.
  - [ ] **8.3** Add optional volatility-scaled slippage logic per `specs.md` §13 (`B-05`).
  - [ ] **8.4** Implement tests in `tests/test_costs.py`: `T-06` (100 HOLD steps under drift yield $0$ cost), `T-07` (hand-computed $10\%$ asset trade matches cost $0.0002$), and `T-16` (volatility-scaled slippage formula bounds).

* **Artifacts Created:** `src/env/portfolio.py`, `src/env/costs.py`, `tests/test_costs.py`
* **Verification Criteria:** Tests `T-06`, `T-07`, and `T-16` pass.

---

### Step 09 — Action Space & Observation Construction
* **Prerequisites:** Steps 06, 07, 08 | **Lane:** B
* **Goal:** Define discrete 31-action rebalancing map and construct continuous observation vector.

* **Sub-tasks:**
  - [ ] **9.1** Build 31-action lookup table ($0=\text{HOLD}$, $1\dots30=\text{move } 10\% \text{ between buckets}$) and export mapping to `artifacts/action_table.json`.
  - [ ] **9.2** Implement observation aggregator concatenating 30 z-scored features, 6 portfolio weights, and 3 regime probabilities into a 39-D vector.
  - [ ] **9.3** Implement feature configuration toggle flags (`include_weights_in_obs`, `regime.enabled`, `regime.include_novelty`).
  - [ ] **9.4** Write test `T-04` in `tests/test_env.py`: confirm action table bijection, action $0$ equality to HOLD, and exclusion of self-trades (`from == to`).

* **Artifacts Created:** `artifacts/action_table.json`, observation assembly methods in `src/env/portfolio_env.py`, `tests/test_env.py`
* **Verification Criteria:** Test `T-04` passes; sample vector observation matches input feature parquet files bitwise.

---

### Step 10 — Gymnasium Environment Construction
* **Prerequisites:** Step 09 | **Lane:** B
* **Goal:** Wrap portfolio engine and state space into a compliant Gymnasium environment interface.

* **Sub-tasks:**
  - [ ] **10.1** Implement `src/env/portfolio_env.py` adhering to standard `reset()` and `step()` Gymnasium signatures (returning 5-element tuples).
  - [ ] **10.2** Implement reset options handler accepting execution window specifications or explicit index boundaries (`start_idx`, `length`).
  - [ ] **10.3** Set environment episode initial conditions: portfolio value $V_0 = 1.0$, equal asset weights ($0.2$ each), zero cash.
  - [ ] **10.4** Populate step step output `info` dict per `specs.md` `E-09`.
  - [ ] **10.5** Implement tests `T-05` (10,000 random actions preserve weight non-negativity and unit sum) and `T-10` (`gymnasium.utils.env_checker.check_env` compliance).

* **Artifacts Created:** `src/env/portfolio_env.py`, updated `tests/test_env.py`
* **Verification Criteria:** Tests `T-05` and `T-10` pass cleanly.

---

### Step 11 — Risk-Sensitive Reward Function
* **Prerequisites:** Step 10 | **Lane:** B
* **Goal:** Implement single-deduction, risk-sensitive reward function incorporating downside volatility and drawdown penalties.

* **Sub-tasks:**
  - [ ] **11.1** Write `src/env/reward.py` evaluating:
        $$\text{reward}_t = 100 \times \left( r_{\text{gross}} - \frac{\text{cost}_t}{V_t} - \lambda_1 \cdot \sigma_{\text{down}} - \lambda_2 \cdot \text{DD} \right)$$
  - [ ] **11.2** Store breakdown of raw reward components inside step output `info` dictionary.
  - [ ] **11.3** Implement test `T-09` in `tests/test_reward.py`: verify exact reward formula output within absolute tolerance $10^{-6}$.
  - [ ] **11.4** Implement test `T-08`: verify an all-HOLD policy matches passive Buy & Hold return within absolute tolerance $10^{-9}$.
  - [ ] **11.5** Write diagnostic suite `src/evaluation/diagnostics.py` measuring mean cash, action entropy, turnover, and % HOLD.

* **Artifacts Created:** `src/env/reward.py`, `src/evaluation/diagnostics.py`, `tests/test_reward.py`
* **Verification Criteria:** Tests `T-08` and `T-09` pass; team code review approves environment step mechanics.

---

### Step 12 — Curriculum Sampler & Split Integrity
* **Prerequisites:** Step 10 | **Lane:** B
* **Goal:** Construct multi-stage training curriculum sampler and enforce temporal isolation checks.

* **Sub-tasks:**
  - [ ] **12.1** Implement `src/agents/curriculum.py` managing 3-stage sampling logic ($40\% / 30\% / 30\%$ step allocations) across `calm`, `covid`, and `rates` windows.
  - [ ] **12.2** Add support for tuning mode (excluding `rates` window) and control `A2` mode (uniform random sampling without curriculum).
  - [ ] **12.3** Append equal-weight basket max drawdown window difficulty table to `reports/regime_report.md`.
  - [ ] **12.4** Implement tests in `tests/test_curriculum.py`: `T-19` (stage boundaries, window mix proportions, exclusion of post-2023 dates) and `T-20` (strict exclusion of test and held-out 2025 dates).

* **Artifacts Created:** `src/agents/curriculum.py`, `tests/test_curriculum.py`, update to `reports/regime_report.md`
* **Verification Criteria:** Tests `T-19` and `T-20` pass cleanly.

---

### Step 13 — Metrics Calculation Engine
* **Prerequisites:** Step 04 | **Lane:** C
* **Goal:** Implement standardized calculation routines for financial, risk, and stability metrics.

* **Sub-tasks:**
  - [ ] **13.1** Write `src/evaluation/metrics.py` computing total return, CAGR, annualized volatility, Sharpe ratio, Sortino ratio, Max Drawdown, Calmar ratio, daily 5% CVaR, and turnover.
  - [ ] **13.2** Implement logic leaving CAGR and Calmar blank (NaN) for evaluation periods under 252 trading days.
  - [ ] **13.3** Implement rolling 21-day metric utility functions.
  - [ ] **13.4** Write test `T-14` in `tests/test_metrics.py`: validate computed metrics against hand-calculated synthetic vectors.

* **Artifacts Created:** `src/evaluation/metrics.py`, `tests/test_metrics.py`
* **Verification Criteria:** Test `T-14` passes.

---

### Step 14 — Baseline Benchmarks (B&H, 1/N, MVO)
* **Prerequisites:** Steps 08, 13 | **Lane:** C
* **Goal:** Build non-RL benchmark strategies using shared execution environment rules and cost models.

* **Sub-tasks:**
  - [ ] **14.1** Implement Buy & Hold strategy `src/benchmarks/buy_hold.py`.
  - [ ] **14.2** Implement periodic 21-day equal-weight rebalancing strategy `src/benchmarks/equal_weight.py`.
  - [ ] **14.3** Implement trailing 252-day Mean-Variance Optimization strategy `src/benchmarks/mvo.py` using SciPy SLSQP with $10^{-6} \cdot I$ ridge regularization.
  - [ ] **14.4** Write test `T-13` in `tests/test_benchmarks.py`: confirm fee parity across benchmark simulator and Gymnasium environment.
  - [ ] **14.5** Dry-run benchmark strategies on `val_2022` window and write output CSVs to `results/validation/val_2022/`.

* **Artifacts Created:** `src/benchmarks/*.py`, `tests/test_benchmarks.py`, baseline CSVs in `results/validation/val_2022/`
* **Verification Criteria:** Test `T-13` passes; benchmark outputs generated for validation window.

---

### Step 15 — Training Infrastructure & Time Budget
* **Prerequisites:** Steps 11, 12, 13
* **Goal:** Establish generic training loop, replay buffers, greedy evaluator, and execute wall-clock benchmark.

* **Sub-tasks:**
  - [ ] **15.1** Write experience replay buffer `src/agents/replay.py`.
  - [ ] **15.2** Construct main agent runner script `src/agents/train.py` handling configuration loading, seed setup, curriculum updates, and training log creation.
  - [ ] **15.3** Create deterministic greedy backtest evaluator `src/evaluation/backtest.py`.
  - [ ] **15.4** Implement execution check in `backtest.py` throwing an exception if evaluation is attempted on test windows without an active freeze tag.
  - [ ] **15.5** Measure wall-clock execution time for 10,000 steps and publish time budget projection to `reports/time_budget.md`.

* **Artifacts Created:** `src/agents/replay.py`, `src/agents/train.py`, `src/evaluation/backtest.py`, `reports/time_budget.md`
* **Verification Criteria:** Benchmark run executes 1,000 steps reproducibly across identical seeds; time budget report is populated.

---

### Step 16 — Standard Deep Q-Network (DQN)
* **Prerequisites:** Step 15
* **Goal:** Implement standard Deep Q-Network agent baseline.

* **Sub-tasks:**
  - [ ] **16.1** Implement PyTorch neural network and agent module `src/agents/dqn.py` (Architecture: `39 -> 128 -> 128 -> 31`, Huber loss, target network update frequency = 500 steps).
  - [ ] **16.2** Write toy MDP unit test in `tests/test_dqn.py` verifying loss minimization on synthetic state inputs.
  - [ ] **16.3** Run 10,000 step sanity verification to confirm action selection diversity and policy convergence over random agent baseline.

* **Artifacts Created:** `src/agents/dqn.py`, `tests/test_dqn.py`
* **Verification Criteria:** Toy MDP loss decreases monotonically; agent action histogram demonstrates non-degenerate distribution.

---

### Step 17 — Reward Penalty Grid Search ($\lambda_1, \lambda_2$)
* **Prerequisites:** Step 16
* **Goal:** Execute tuning grid search for risk penalty coefficients using validation data.

* **Sub-tasks:**
  - [ ] **17.1** Configure grid search parameter space: $C_0=(0, 0)$, $C_1=(0.05, 0.002)$, $C_2=(0.10, 0.005)$.
  - [ ] **17.2** Train DQN agent over 2 seeds $\times$ 75,000 steps per grid cell using tuning pool (`2012-2021`).
  - [ ] **17.3** Evaluate trained models on `val_2022` and filter out runs violating non-degeneracy criteria (`RW-04`).
  - [ ] **17.4** Select cell with highest median Sortino ratio, update `config/config.yaml` with selected $\lambda_1, \lambda_2$, and record selection in `results/tuning/lambda_grid.csv`.

* **Artifacts Created:** `results/tuning/lambda_grid.csv`, updated `config/config.yaml`
* **Verification Criteria:** Parameters $\lambda_1$ and $\lambda_2$ in `config.yaml` are updated from `null` to chosen float values.

---

### Step 18 — Compressed-State Tabular Q-Learner
* **Prerequisites:** Steps 15, 17
* **Goal:** Construct tabular Q-learning baseline using discretized market feature mapping.

* **Sub-tasks:**
  - [ ] **18.1** Create state compression module mapping continuous features to 135 discrete states ($\text{Regime}[3] \times \text{Trend}[3] \times \text{Leader}[5] \times \text{Cash}[3]$).
  - [ ] **18.2** Fit trend tercile discretization bounds strictly on training fit window and export bounds to `artifacts/tabular_cuts.json`.
  - [ ] **18.3** Write `src/agents/tabular_q.py` ($\alpha=0.1, \gamma=0.99$) using curriculum sampler.
  - [ ] **18.4** Implement unit test verifying deterministic mapping and reachability across all 135 state indexes.

* **Artifacts Created:** `src/agents/tabular_q.py`, `artifacts/tabular_cuts.json`, unit test in `tests/test_tabular.py`
* **Verification Criteria:** State mapping unit test passes; tabular agent completes short training run without index errors.

---

### Step 19 — Distributional Quantile RL (QR-DQN)
* **Prerequisites:** Steps 16, 17
* **Goal:** Implement Quantile Regression DQN with CVaR action selection.

* **Sub-tasks:**
  - [ ] **19.1** Implement `src/agents/qrdqn.py` outputting $N=20$ quantiles per action ($31 \times 20$ output head).
  - [ ] **19.2** Implement Quantile Huber Loss across all 20 quantiles; assert internal tensor shape `[Batch, N, N]`.
  - [ ] **19.3** Implement CVaR action selection: sort quantiles ascending, average lowest $k=\lceil \alpha N \rceil$ values ($k=4$ for $\alpha=0.2$). Apply CVaR operator to both target action selection and greedy action execution.
  - [ ] **19.4** Write test `T-11` in `tests/test_qrdqn.py`: validate learned quantiles against analytical normal distribution inverse CDF on a single-state MDP.
  - [ ] **19.5** Write test `T-12`: verify sorting requirement and quantile slice indexing for $\alpha=1.0$ ($k=20$) and $\alpha=0.2$ ($k=4$).

* **Artifacts Created:** `src/agents/qrdqn.py`, `tests/test_qrdqn.py`
* **Verification Criteria:** Tests `T-11` and `T-12` pass cleanly.

---

### Step 20 — Validation Scoring & Headline Selection
* **Prerequisites:** Steps 18, 19
* **Goal:** Rank candidate models on validation period `val_2022` and nominate primary headline strategy.

* **Sub-tasks:**
  - [ ] **20.1** Train candidate architectures (`tabular_q`, `dqn`, `qrdqn_a1.0`, `qrdqn_a0.2`) across 2 seeds $\times$ 150,000 steps on tuning pool (`2012-2021`).
  - [ ] **20.2** Execute greedy evaluation passes over `val_2022` window and record output to `results/tuning/validation_comparison.csv`.
  - [ ] **20.3** Filter out runs failing non-degeneracy guard (`RW-04`).
  - [ ] **20.4** Apply headline selection tie-breaker hierarchy (`TU-04`) to select winning model architecture; record nomination to `results/tuning/headline.json`.

* **Artifacts Created:** `results/tuning/validation_comparison.csv`, `results/tuning/headline.json`
* **Verification Criteria:** All 8 tuning runs are logged; single headline strategy is saved to `headline.json`.

---

### Step 21 — Full Curriculum Training Run
* **Prerequisites:** Step 20
* **Goal:** Train final model instances and experimental controls across full training set (`2012-2023`).

* **Sub-tasks:**
  - [ ] **21.1** Train full strategy suite (`tabular_q`, `dqn`, `qrdqn_a1.0`, `qrdqn_a0.2`, `qrdqn_a0.2_noregime`, `qrdqn_a0.2_nocurr`) across 3 seeds ($0, 1, 2$) $\times$ 150,000 steps using fit window `2012-2023`.
  - [ ] **21.2** Save network weights and Q-tables to `artifacts/models/{strategy}_seed{k}.pt` embedding config hash into model header metadata.
  - [ ] **21.3** Output step training logs and degeneracy metrics to `results/training/{strategy}_seed{k}.csv`.

* **Artifacts Created:** Checkpoint artifacts in `artifacts/models/`, logs in `results/training/`
* **Verification Criteria:** All 18 target checkpoint files exist and contain valid embedded config hashes.

---

### Step 22 — Configuration & Codebase Freeze
* **Prerequisites:** Steps 14, 21
* **Goal:** Hash configuration state and lock project codebase prior to unblinding test metrics.

* **Sub-tasks:**
  - [ ] **22.1** Run full verification test suite (`pytest` for `T-01` through `T-14`, `T-16`, `T-18` through `T-20`); confirm 100% pass rate.
  - [ ] **22.2** Write script `scripts/freeze_config.py` computing SHA-256 hash of `config/config.yaml` and writing output to `config/FROZEN`.
  - [ ] **22.3** Execute git tag command `git tag config-frozen`.

* **Artifacts Created:** `scripts/freeze_config.py`, `config/FROZEN`, Git tag `config-frozen`
* **Verification Criteria:** File `config/FROZEN` exists; git repository contains `config-frozen` tag.

---

### Step 23 — Blind Backtest Execution
* **Prerequisites:** Step 22
* **Goal:** Execute single greedy evaluation pass across test windows (`test_normal` and `test_war`).

* **Sub-tasks:**
  - [ ] **23.1** Write backtest runner script `scripts/run_final_backtest.py` incorporating validation guard test `T-15` (verifying active `config/FROZEN` match).
  - [ ] **23.2** Run all trained model checkpoints and benchmarks across `test_normal` (2024) and `test_war` (`2026-02-27` to `2026-04-30`) with zero exploration ($\varepsilon=0$) and fixed weights.
  - [ ] **23.3** Write individual run evaluation logs to `results/backtests/{scenario}/{strategy}_seed{k}.csv` following 14-column output format.
  - [ ] **23.4** Compute cross-seed average metrics and export `{strategy}_mean.csv` files.

* **Artifacts Created:** `scripts/run_final_backtest.py`, test backtest CSV files in `results/backtests/test_normal/` and `results/backtests/test_war/`
* **Verification Criteria:** Test `T-15` passes; all backtest CSVs exist, contain non-NaN values, start at $V_0=1.0$, and match scenario index bounds.

---

### Step 24 — Post-Freeze Diagnostic & Slippage Analysis
* **Prerequisites:** Step 23
* **Goal:** Compute scenario breakdown statistics, regime mapping overlay, and slippage sensitivity runs.

* **Sub-tasks:**
  - [ ] **24.1** Write `src/evaluation/scenarios.py` assigning phase labels (`normal`, `war_drawdown`, `war_recovery`) and computing `high_vol` indicator flags.
  - [ ] **24.2** Compute realized correlation spikes, volatility ratios, and feature shift metrics across test periods; append findings to `reports/regime_report.md`.
  - [ ] **24.3** Re-run backtest evaluation with volatility-scaled slippage enabled (`slippage_mode: vol_scaled`) saving outputs to `results/backtests_slippage/{scenario}/`.
  - [ ] **24.4** Consolidate strategy performance metrics across scenarios, phases, and slippage modes into master file `results/metrics_summary.csv`.

* **Artifacts Created:** `src/evaluation/scenarios.py`, `results/backtests_slippage/`, `results/metrics_summary.csv`, update to `reports/regime_report.md`
* **Verification Criteria:** Master summary file `results/metrics_summary.csv` contains complete data rows for all strategy/seed/phase combinations.

---

### Step 25 — Success Criteria & Hypothesis Verdicts
* **Prerequisites:** Step 24
* **Goal:** Formally evaluate pre-registered success criteria `SC-1`–`SC-4` and test hypotheses `H1`–`H5`.

* **Sub-tasks:**
  - [ ] **25.1** Evaluate headline model performance against success criteria `SC-1`, `SC-2`, `SC-3`, and `SC-4`; write formal result object to `results/verdict.json`.
  - [ ] **25.2** Evaluate support criteria for research hypotheses `H1` through `H5`.
  - [ ] **25.3** Compute defensive allocation statistics (cash allocation at days 5, 10, 20) during `test_war` window.
  - [ ] **25.4** Write comprehensive stress test report `reports/stress_test.md` containing performance tables, weight breakdown plots, and limitation disclosures `L-01` through `L-11`.

* **Artifacts Created:** `results/verdict.json`, `reports/stress_test.md`
* **Verification Criteria:** File `results/verdict.json` correctly exports overall boolean status (`"satisfactory": true/false`); stress report is populated.

---

### Step 26 — Result Manifest & Data Export
* **Prerequisites:** Step 24
* **Goal:** Format backtest datasets to full 18-column standard and build result manifest file.

* **Sub-tasks:**
  - [ ] **26.1** Create export utility `src/evaluation/export.py` appending `regime`, `drawdown`, `rolling_sharpe_21`, and `rolling_sortino_21` columns to all backtest CSVs (yielding full 18-column `C-01` schema).
  - [ ] **26.2** Generate master project manifest `results/manifest.json` recording config hash, git commit, creation timestamp, and file checksums.

* **Artifacts Created:** `src/evaluation/export.py`, updated 18-column backtest CSVs, `results/manifest.json`
* **Verification Criteria:** All CSV files in `results/backtests/` contain exactly 18 columns in spec-defined order (`C-01`).

---

### Step 27 — Streamlit Core Architecture & Data Loaders
* **Prerequisites:** Step 26
* **Goal:** Build resilient data loading layer and UI base configuration for interactive dashboard.

* **Sub-tasks:**
  - [ ] **27.1** Implement `dashboard/loaders.py` using `@st.cache_data` to load metrics summaries, backtest CSVs, and verdict metadata.
  - [ ] **27.2** Implement schema validation checks inside data loader throwing user-friendly error messages on missing or corrupted files.
  - [ ] **27.3** Create primary app entry point `dashboard/app.py` featuring sidebar navigation, scenario selectbox, strategy multiselect, and slippage toggle.
  - [ ] **27.4** Write test `T-17` in `tests/test_dashboard.py`: verify dashboard data loader handles missing files without raising uncaught exceptions.

* **Artifacts Created:** `dashboard/app.py`, `dashboard/loaders.py`, `tests/test_dashboard.py`
* **Verification Criteria:** Test `T-17` passes; main Streamlit application starts successfully.

---

### Step 28 — Streamlit Interactive Views
* **Prerequisites:** Step 27
* **Goal:** Build visualization components and interactive risk tear-sheet views.

* **Sub-tasks:**
  - [ ] **28.1** Construct cumulative performance equity curves chart (with log scale toggle).
  - [ ] **28.2** Construct rolling 21-day Sharpe and Sortino ratio plots.
  - [ ] **28.3** Construct portfolio drawdown profile chart.
  - [ ] **28.4** Implement executive risk tear-sheet displaying metrics table and `SC-1`–`SC-4` pass/fail status from `verdict.json`.
  - [ ] **28.5** Construct asset allocation stacked area chart showing seed dispersion bands.
  - [ ] **28.6** Construct market regime probability timeline plot ($p_{\text{calm}}, p_{\text{choppy}}, p_{\text{crisis}}$).

* **Artifacts Created:** Visualizations within `dashboard/app.py`
* **Verification Criteria:** Dashboard views render smoothly using cached static backtest datasets.

---

### Step 29 — Streamlit Error-Handling Audit
* **Prerequisites:** Step 28
* **Goal:** Audit dashboard interface stability under missing data and edge configuration states.

* **Sub-tasks:**
  - [ ] **29.1** Test dashboard display behavior when individual backtest CSV files are removed.
  - [ ] **29.2** Test dashboard rendering under single-strategy selection and empty selection bounds.
  - [ ] **29.3** Verify application presents clear explanatory error notices rather than Python stack traces during failure states.

* **Artifacts Created:** UI error handling updates in `dashboard/loaders.py`
* **Verification Criteria:** Dashboard displays clear warning banners for missing inputs without crashing.

---

### Step 30 — Reproduction Test Suite & Execution Scripts
* **Prerequisites:** Step 29
* **Goal:** Finalize root documentation and verify full pipeline execution from a clean environment clone.

* **Sub-tasks:**
  - [ ] **30.1** Write main repository `README.md` detailing setup instructions, module overview, and workflow execution order.
  - [ ] **30.2** Populate `Makefile` targets (`all`, `data`, `features`, `regime`, `train`, `freeze`, `backtest`, `dashboard`).
  - [ ] **30.3** Execute end-to-end pipeline run from a fresh repository clone using `make all`.

* **Artifacts Created:** `README.md`, updated `Makefile`
* **Verification Criteria:** End-to-end execution `make all` finishes successfully on a clean clone, regenerating all results datasets.

---

### Step 31 — Final Audit & Project Report
* **Prerequisites:** Steps 25, 30
* **Goal:** Write comprehensive final project report and verify data traceability.

* **Sub-tasks:**
  - [ ] **31.1** Write `reports/final_report.md` synthesizing methodology, results summary, hypothesis verdicts, limitations (`L-01`–`L-11`), and decision logs.
  - [ ] **31.2** Perform full data audit ensuring every metric reported in `final_report.md` matches `results/manifest.json`.
  - [ ] **31.3** Update status board to reflect 100% step completion.

* **Artifacts Created:** `reports/final_report.md`
* **Verification Criteria:** Final report audit succeeds with zero unreferenced or mismatched numerical values.

---

## 4. Interrupt & Session Tracking Log

| Date | Contributor | Task ID / Sub-task | Action Taken / Changes Made | Commit Hash | Next Step ID |
|---|---|---|---|---|---|
| 2026-11-06 | Team | Step 00 | Established initial specifications (`specs.md` v1.1) and master plan. | — | Step 01 |
| 2026-11-06 | Copilot assistant | Step 02 (2.1–2.6) | Added the §20 scaffold, §19 config, pinned requirements, seed utility, T-01–T-20 xfail placeholders, and Makefile; verified 20 xfailed in an isolated virtualenv. Step 01 is now complete. | — | Step 03 |
| 2026-11-06 | Copilot assistant | Step 03 (3.1–3.4) | Added the yfinance probe and adjusted-OHLCV downloader with verified offline caching, refresh support, and metadata writing. Upgraded and pinned yfinance 1.8.0; downloaded and cached 3,602 rows (2012-01-03 through 2026-04-30). | — | Step 04 |
| 2026-11-06 | Copilot assistant | Step 04 (4.1–4.5) | Added D-04–D-06 cleaning, calendar alignment, fill auditing, T-01 sanity coverage, processed parquet, and data-quality report. Processed 3,602 rows with zero fills, zero non-finite values, and no unexplained >25% return. Tests: 18 passed, 19 xfailed. | — | Step 05 |
| 2026-11-06 | Copilot assistant | Step 05 (5.1–5.6) | Added causal stock indicators and market features, warm-up-trimmed feature pipeline, metadata, and T-02 causality coverage at 50 seeded cut dates. Generated aligned feature outputs with 3,553 rows from 2012-03-14 and no NaN/inf. Tests: 24 passed, 18 xfailed. | — | Step 06 |
| 2026-11-06 | Copilot assistant | Step 06 (6.1–6.5) | Added reusable fit-window scaler with persisted load/transform parameters; generated final (2012–2023) and tuning (2012–2021) scalers and normalized full-series parquets. T-03 verifies no post-window fitting. Full suite: 28 passed, 17 xfailed. Fit-window moments are near zero/unit variance before required ±5 clipping. | — | Step 07 |
