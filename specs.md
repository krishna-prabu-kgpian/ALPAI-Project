# specs.md — Risk-Sensitive RL Portfolio Allocation (DQN → QR-DQN) under Transaction Costs & Regime Stress

| | |
|---|---|
| **Version** | 1.0 (draft — needs team acknowledgement, see §19) |
| **Date** | 2026-11-06 |
| **Companion file** | `TODO.md` (task tracker; references the rule IDs defined here) |
| **Authority** | This file is the **single source of truth**. If code, a notebook, a chat message or an AI answer disagrees with it, **this file wins** — or you amend it first (§19). |

---

## 0. Ground Rules (read before touching code)

| ID | Rule |
|---|---|
| GR-1 | **Spec wins.** Never "fix" a discrepancy silently in code. Either the code is wrong or the spec needs a logged amendment. |
| GR-2 | **Unspecified = unknown.** If this file doesn't define something, do not invent it. Add an Open Question (§18.2) and ask the team. |
| GR-3 | **No magic numbers.** Every constant lives in `config/config.yaml` (skeleton in §16) and mirrors §3. |
| GR-4 | **Hypotheses are not results.** "DQN collapses in shocks" and "QR-DQN shifts defensive before a crash" are hypotheses (H1–H3) until a committed script produces evidence. Negative results are valid and must be reported. |
| GR-5 | **Verify library APIs against the installed version** (`yfinance`, `gymnasium`, `torch` change often). Run a 3-line probe or read `help()`. Never trust a remembered signature or column name. |
| GR-6 | **Every reported number is traceable**: committed script + config hash, recorded in `results/manifest.json`. |
| GR-7 | **The test set is blind** (S-03). |
| GR-8 | **AI-assisted code must pass the tests in §14 before merge.** If an AI suggests changing a constant or formula from this file, treat it as a spec amendment (§19), not a code tweak. |
| GR-9 | **"Done" = Done-when criteria in `TODO.md` met + tests green + `TODO.md` updated in the same commit.** |

---

## 1. Goal, Hypotheses, Scope

**Research question.** Does a distributional, tail-risk-aware RL agent (QR-DQN with CVaR action selection) allocate a 5-stock portfolio better — in risk-adjusted terms, net of 10 bps costs — than a mean-based DQN, a tabular Q-learner and classical benchmarks, especially in stress regimes absent from its training data?

**Hypotheses (to be tested, not assumed):**
- **H1** — Standard DQN suffers deeper drawdowns / worse tail losses than QR-DQN in shock windows.
- **H2** — QR-DQN with CVaR (α = 0.2) has lower max drawdown and CVaR than QR-DQN with α = 1.0 (risk-neutral) at similar or modestly lower return. This isolates the effect of distributional risk-sensitivity from the reward penalty.
- **H3** — If RL agents beat the benchmarks, it will show up in drawdown / tail metrics, not raw return.

**Non-goals.** No shorting, leverage, derivatives, intraday data, live trading, tax modeling, or claims of real-world profitability.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| `A` | Assets, **fixed order everywhere**: `[NVDA, AAPL, MSFT, GOOGL, AMZN]` |
| `B` | Buckets = `A + [CASH]` (6 entries, CASH is index 5) |
| `t` | Trading day index (close of day `t`) |
| `P_{i,t}` | Adjusted close of asset `i` on day `t` |
| `r_{i,t+1}` | Simple return `P_{i,t+1}/P_{i,t} − 1` (cash: 0) |
| `V_t` | Portfolio value at close of `t` (before trading on `t`) |
| `w_t^pre` | Drifted weights at close of `t`, before trading (6-vector, sums to 1) |
| `w_t^tgt` | Target weights after applying action `a_t` (6-vector, sums to 1) |
| `c` | Proportional transaction cost = 0.0010 (10 bps) |

---

## 3. Global Constants (mirror in `config.yaml`)

| Parameter | Value |
|---|---|
| Universe / order | `NVDA, AAPL, MSFT, GOOGL, AMZN` |
| Frequency | Daily |
| Download window | start `2012-01-01`, end `2024-01-01` (**exclusive**; yields data through 2023-12-29) |
| Train (in-sample) | 2012-01-01 → 2019-12-31 |
| Tuning split (inside train) | tune-train 2012-01-01 → 2017-12-31, validation 2018-01-01 → 2019-12-31 |
| Test (out-of-sample, blind) | 2020-01-01 → 2023-12-31 |
| Transaction cost `c` | 0.0010 on traded notional of **risky assets** |
| Rebalance step | 0.10 |
| Initial portfolio | `V0 = 1.0`, equal weights 0.2 × 5, cash 0 (all strategies) |
| Cash yield | 0 % |
| Annualization / risk-free | 252 days / 0 |
| Feature windows | SMA 10, SMA 50, RSI 14, vol 20, volume 20 |
| z-score clip | ±5 |
| Discount `γ` | 0.99 |
| Seeds | `0,1,2,3,4` |

---

## 4. Data Spec (Task 1.1)

| ID | Requirement |
|---|---|
| D-01 | **Call:** `yfinance.download(tickers, start="2012-01-01", end="2024-01-01", interval="1d", auto_adjust=True, progress=False)`. `end` is exclusive. Expect first row 2012-01-03, last row 2023-12-29. |
| D-02 | **Adjustment consistency.** `auto_adjust=True` returns split- *and* dividend-adjusted O/H/L/C (no separate "Adj Close" column). Never combine an adjusted close with unadjusted O/H/L. Internal column `close` = adjusted close. Probe the returned structure first (multi-ticker downloads return MultiIndex columns in current versions). |
| D-03 | **Cache raw data** to `data/raw/prices_raw.parquet` + `prices_raw.meta.json` (UTC timestamp, yfinance version, row counts, SHA-256). All downstream code reads the cache. Re-download only via an explicit `--refresh` flag **and** a Decision Log entry (Yahoo data can be revised retroactively). |
| D-04 | **Calendar.** All five tickers share one exchange calendar → assert identical date indexes. If not, reindex to the union and apply D-05. |
| D-05 | **Missing data.** `ffill(limit=5)` on price columns only. **Never `bfill`, never interpolate** (both use future information). Missing volume → 0. Log the number of filled cells; if > 0.1 % of cells, stop and investigate. After the indicator warm-up period, assert zero NaN/inf. |
| D-06 | **Sanity checks:** strictly increasing unique `DatetimeIndex`; prices > 0; volume ≥ 0; row count in 3,000–3,030; print the 10 largest absolute daily returns per asset — any |ret| > 25 % must be explained (e.g. an unadjusted split) before proceeding. |
| D-07 | **Outputs:** `data/processed/prices_clean.parquet`, `reports/data_quality.md`. |

---

## 5. Split Spec

| ID | Requirement |
|---|---|
| S-01 | Train = 2012–2019 (in-sample). Test = 2020–2023 (out-of-sample). |
| S-02 | **Hyperparameter tuning uses only a validation slice carved from train** (tune-train 2012–2017 / validation 2018–2019). After choosing hyperparameters, retrain on the full 2012–2019 window. *(The original roadmap said "do not retune on test" but defined no validation set; this closes that gap.)* |
| S-03 | **Test-set guard.** Test-period metrics may only be produced by `scripts/run_final_backtest.py`, which refuses to run unless `config/FROZEN` contains the hash of the current `config.yaml` (T-15). Any test metric computed before freezing contaminates the test set; log it in the Decision Log and disclose it. |
| S-04 | Features need up to 50 days of history. Warm-up rows at the start of the series are dropped (training effectively begins ≈ mid-March 2012). Test features are computed on the continuous series, so 2020-01-02 already has full history (see F-01). |

---

## 6. Feature Spec (Tasks 1.2 & 1.3)

**F-01 — What "no look-ahead" actually means here (clarifies the roadmap).**
- ✅ **Allowed and required:** *causal* rolling windows (trailing only) computed on the **continuous** 2012–2023 series, then split by date. If you compute test features separately, the first ~50 test days become NaN or wrong.
- ❌ **Look-ahead bugs:** centered windows, `bfill`, `shift(-k)`, normalization statistics fit on anything beyond the training window, off-by-one date alignment.

**F-02 — Six features per asset (fixed order; stationary by construction).**

| # | Name | Definition |
|---|---|---|
| 0 | `ret_1d` | `log(P_t / P_{t-1})` |
| 1 | `sma10_gap` | `P_t / SMA10_t − 1` |
| 2 | `sma50_gap` | `P_t / SMA50_t − 1` |
| 3 | `rsi14` | Wilder RSI: `ewm(alpha=1/14, adjust=False, min_periods=14)` on gains/losses. If avg loss = 0 → RSI = 100 (and 50 if avg gain = 0 too). |
| 4 | `vol20` | `std(ret_1d, 20, ddof=1) × sqrt(252)` |
| 5 | `vol_chg` | `log((Volume_t + 1) / (mean(Volume over last 20 days incl. t) + 1))` |

*Raw SMAs are non-stationary price levels, so they enter only as price-relative gaps. The roadmap listed 4 indicators but demands a 5×6 observation, so `ret_1d` and `vol_chg` were added (DL-01).*

**F-03 — Returns** always use the adjusted close.

**F-04 — Normalization.**
`z = (x − μ_fit) / max(σ_fit, 1e-8)`, per (asset, feature) → 30 means, 30 stds. Then clip to `[−5, +5]` (fixed bounds, not data-derived).
- `μ_fit, σ_fit` come from the **training window only**: 2012–2019 for final models; 2012–2017 for tuning runs.
- Persist to `artifacts/scaler_{fit_window}.json`. The test path **loads** the scaler; it never refits.
- Test-set z-scores will *not* have mean 0 / std 1 (2020 volatility is out-of-distribution). That is correct, not a bug.

**F-05 — Causality test (must pass, T-02).** For random cut dates `t`, features at `t` computed from `data[: t+1]` must equal features at `t` computed from the full dataset.

**F-06 — Layout.** Columns are a MultiIndex `(asset, feature)`. The flattened observation is **asset-major**: `obs[asset_idx*6 + feature_idx]`. Outputs: `data/processed/features_raw.parquet`, `data/processed/features_z_{fit_window}.parquet`.

---

## 7. Environment Spec (Phase 2)

| ID | Requirement |
|---|---|
| E-01 | **Gymnasium API (not legacy `gym`).** `reset(seed=None, options=None) -> (obs, info)`; `step(a) -> (obs, reward, terminated, truncated, info)` (5-tuple). Must pass `gymnasium.utils.env_checker.check_env` (T-10). |
| E-02 | **Time & execution convention** (the section most likely to be implemented wrongly): see below. |
| E-03 | **Costs apply only to trades**, never to passive drift (see step 3–4 below). |
| E-04 | **Observation:** `Box(shape=(36,), float32)`. `[0:30]` = z-scored market features (F-06 order). `[30:36]` = current drifted weights `w_t^pre` in order `[NVDA, AAPL, MSFT, GOOGL, AMZN, CASH]`. |
| E-05 | **Action space:** `Discrete(31)`. `0` = HOLD. `1..30` = ordered pairs `(from, to)`, `from ≠ to`, over the 6 buckets, enumerated row-major. Decode table built once, saved to `artifacts/action_table.json`, bijection-tested (T-04). |
| E-06 | **Action semantics:** move `min(0.10, w_pre[from])` of portfolio value from bucket `from` to bucket `to`. If `w_pre[from] < 1e-9` the action is a no-op (cost 0). Long-only, no leverage. |
| E-07 | **Invariants after every step:** all weights ≥ 0 and `|Σw − 1| < 1e-9`. Assert in debug mode (T-05). |
| E-08 | **Episodes.** *Training:* random start, fixed length `L = 252` days, sampled uniformly from the train window. *Evaluation:* one deterministic pass over the entire window from its first day, ε = 0. `truncated=True` at the last day with a valid `t+1`; `terminated` is never True (no leverage → no ruin). Running peak (drawdown) and downside window reset at episode start. |
| E-09 | **Info dict** per step: `date, V, cost, turnover, weights (6), r_gross, r_net, drawdown, sigma_down, action_id`. The backtest logger consumes this — nothing is recomputed elsewhere. |

**E-02 / E-03 — Order of operations inside `step(a_t)`:**

1. Agent observes `obs_t` (features through the close of day `t`, plus `w_t^pre`).
2. Decode `a_t` → target weights `w_t^tgt` (E-06).
3. **Traded notional** per risky asset: `|w_t^tgt,i − w_t^pre,i| × V_t`. *(Measured against the **drifted** weights, so a HOLD action costs exactly zero even though weights drift passively.)*
4. **Cost** = `c × Σ_{i ∈ assets} traded_i` (CASH is not charged). Moving 10 % A→B pays on both legs (0.2 × c × V); cash↔asset pays one leg. Optional slippage (B-05) is added here.
5. `V_post = V_t − cost`. Costs are deducted proportionally from all holdings, so post-trade weights equal `w_t^tgt` exactly (invariant E-07 preserved).
6. Hold to close of `t+1`: each holding grows by `(1 + r_{i,t+1})`, cash by 1. `V_{t+1} = Σ holdings`; `w_{t+1}^pre` = holdings / `V_{t+1}`.
7. Compute the reward (§8) and return `obs_{t+1}`.

**Execution assumption (A-01, DL-07).** Trades fill at the close price of day `t` — the same close that fed the features (market-on-close). This is the standard vectorized-backtest convention. Optional robustness check: `execution: next_open` (fill at `Open_{t+1}`, hold to `Open_{t+2}`). The headline results use `close`.

**Cash (A-02).** Earns 0 %.

---

## 8. Reward Spec (Task 2.3)

**RW-01 — Formula (cost counted exactly once):**

```
reward_t = reward_scale × ( r_gross − cost_t / V_t − λ1 · σ_down − λ2 · DD )
```

- `r_gross = Σ_i w_t^tgt,i · r_{i,t+1}` — return of the post-trade portfolio, **before** cost.
- `cost_t / V_t` — the cost fraction from step 4. *The roadmap formula `R_portfolio − … − Cost` double-counts cost if `R_portfolio` is already net of cost. Here `r_gross` is explicitly gross.* Portfolio value `V` and all metrics use the net path; the difference to the reward is second-order (T-09 tolerance 1e-6).
- `σ_down` — trailing downside semi-deviation of **net daily portfolio returns in the current episode**: `sqrt(mean(min(r_net, 0)²))` over the last ≤ 20 returns including the current one, MAR = 0, daily units (not annualized). Fewer than 2 returns available → 0.
- `DD = (peak_V − V_{t+1}) / peak_V ≥ 0` with `peak_V` the running maximum within the episode.
- `reward_scale = 100` (reward in percentage points, for network stability). λ's are defined **before** scaling.

**RW-02 — Scale warning (why "tune empirically" needs a sensible grid).** Daily mean returns are ≈ 5e-4, while the daily downside deviation of a stock portfolio is ≈ 7e-3. A λ1 of 1 would make holding stocks a guaranteed negative reward → the agent sits in cash. Search grid:
- `λ1 ∈ {0, 0.05, 0.1, 0.25}`
- `λ2 ∈ {0, 0.002, 0.005, 0.01}`

**RW-03 — Tuning protocol.** Tune once, on the validation slice (S-02), using the **standard DQN** (so the reward isn't tuned to favor the proposed model), 2 seeds per cell. Select the cell with the best validation Sortino **subject to the degeneracy guard** (RW-04). Freeze `λ1, λ2` for **all** RL agents and record them in the Decision Log.

**RW-04 — Degeneracy guard.** Reject any configuration where, on validation: mean cash weight > 0.80, or HOLD is chosen > 95 % of steps, or action entropy ≈ 0 (the "never trades" collapse), or the agent churns (mean daily turnover > 0.5). Always log these four diagnostics for every training run.

**RW-05 — Sanity identity.** An always-HOLD policy must reproduce the Buy-and-Hold equity curve exactly (T-08).

---

## 9. Model Spec (Phase 3)

**M-01 — Shared training defaults** (starting points; changes only via validation + Decision Log):

| Parameter | Value |
|---|---|
| Network | MLP, hidden `[128, 128]`, ReLU, float32, CPU by default |
| Optimizer | Adam, lr `1e-4`, grad-clip norm 10 |
| Batch / replay | 64 / 100,000 (uniform), learning starts after 1,000 steps |
| Target network | hard update every 500 gradient steps |
| Exploration | ε-greedy, 1.0 → 0.05 linear over first 30 % of steps; evaluation uses ε = 0 |
| Training length | 200,000 env steps (number of steps picked on validation; final retrain uses that fixed number) |
| Seeds | 0–4 for every model; report all |

**M-02 — Tabular Q-learning.** The 36-D continuous observation **cannot** be discretized directly (curse of dimensionality). The tabular agent uses a compressed state, 3 × 3 × 5 × 3 = **135 states**:
1. `vol_regime` (3 bins): equal-weight mean of `vol20` across assets, cut at train-set terciles.
2. `trend_regime` (3 bins): mean of `sma50_gap`, cut at train-set terciles.
3. `leader` (5): index of the asset with the highest `sma50_gap`.
4. `cash_bucket` (3): `w_cash` in `<0.05`, `0.05–0.5`, `≥0.5`.

Cut-points come from the training window only. Q-table zero-initialised, `α = 0.1`, same 31 actions, same γ and ε schedule. The state function is pure, deterministic and unit-tested.

**M-03 — Standard DQN.** `Q(s, ·)` → 31 outputs. Huber loss (κ = 1) on `r + γ(1 − done) · max_a' Q_target(s', a')`. Plain DQN, **not** Double DQN, per the roadmap (`double_dqn: false` flag allowed for ablation).

**M-04 — QR-DQN.**
- Output head: `31 × N`, `N = 20`. Quantile midpoints `τ̂_i = (2i − 1) / (2N)`, `i = 1..N` (0.025, 0.075, …, 0.975).
- **Loss trains ALL N quantiles** (the full distribution): `u_ij = r + γ·θ_j^target(s', a*) − θ_i(s, a)`, quantile Huber `ρ_τ̂_i^κ(u) = |τ̂_i − 1{u<0}| · L_κ(u)` with `L_κ(u) = ½u²` if `|u| ≤ κ` else `κ(|u| − ½κ)`, `κ = 1`. Sum over `i`, mean over `j`, mean over batch. Assert tensor shape `[B, N, N]`.
- **Risk distortion lives in action selection, not in the loss (DL-06).** The roadmap says to wire `τ ∈ [0, 0.2]` into the loss. Training only the lower quantiles would fit only the left tail and bias the learned distribution. The correct pattern: train everything, then act on the lower tail:
  `Q_α(s, a) = mean of the k lowest quantiles`, `k = max(1, ceil(round(α·N, 8)))` → for `α = 0.2, N = 20`: **k = 4**. `α = 1.0` ⇒ plain mean (risk-neutral).
- `Q_α` is used for **both** acting **and** choosing `a*` for the TD target (consistency).
- **Always sort the N quantile outputs ascending before slicing** (quantile crossing is common; unsorted slicing silently averages the wrong quantiles).

**M-05 — Model matrix (minimum):** `Tabular-Q`, `DQN`, `QR-DQN(α=1.0)`, `QR-DQN(α=0.2)`. Optional: α ∈ {0.1, 0.5}; λ = 0 variants.

**M-06 — Artifacts.** `artifacts/models/{model}_seed{k}.pt`, `results/training/{model}_seed{k}.csv` (reward, loss, val metrics, the four RW-04 diagnostics), config hash stored in each checkpoint.

**M-07 — Reproducibility.** Seed `numpy`, `torch`, `random`, and `env.reset(seed=...)`. Evaluation is deterministic.

---

## 10. Backtest & Stress Spec (Phase 4)

| ID | Requirement |
|---|---|
| B-01 | **Final evaluation** runs after `config/FROZEN` is written (S-03). Each agent × each of the 5 seeds makes one greedy pass over 2020-01-02 → 2023-12-29, from the initial portfolio (equal weights). Report mean ± std across seeds plus all individual seeds. **Never select the "best seed" using test results.** |
| B-02 | No retuning and no online learning on the test set. Network weights are frozen. |
| B-03 | **Regime windows are fixed here, before any agent results are seen** (changing them requires a Decision Log entry made *before* looking at results). Labels: `shock_covid` = 2020-02-19 → 2020-03-23; `shock_rates` = 2022-01-03 → 2022-10-12; `normal` = all other test days. Task 4.2.1 sanity-checks these against the basket's own drawdown. Note: the Feb-2022 geopolitical shock lies inside `shock_rates` — war and rate effects **cannot be separated** with this data; report as "2022 macro/geopolitical stress". |
| B-04 | The roadmap's claims "correlations spike to 1.0, volatility triples" are **hypotheses to measure**, not assumptions. Task 4.2.1 reports the observed mean pairwise 20-day correlation and basket-vol ratio vs the train median, per regime. A data-driven flag `high_vol` (basket `vol20` > 2 × train median) is reported alongside the calendar windows. |
| B-05 | **Slippage layer (stress only).** `slippage_mode: vol_scaled`: `slip_bps_{i,t} = s0 × clip(vol20_{i,t} / median_train(vol20_i), 1, 5)`, `s0 = 2 bps`, charged on traded notional, computed causally. Applied **identically to every strategy**. Headline tables: slippage **off** (pure 10 bps, as specified). Stress tables: both modes. T-16 checks that slippage in shock windows exceeds calm-window slippage. |
| B-06 | **Defensive-behaviour analysis.** Agents have no foresight, so "shifted defensive before the crash" is *measured*, not required: report mean cash weight and mean portfolio `vol20` in the 20 trading days before each shock window and during it, plus in-window max drawdown vs Buy-and-Hold and 1/N. If the agent does not de-risk, say so (GR-4). |
| B-07 | Backtest outputs follow the data contracts in §13. |

---

## 11. Metrics Spec (single implementation in `src/evaluation/metrics.py`, used by exporter **and** dashboard)

| ID | Metric | Definition |
|---|---|---|
| X-01 | Total return, CAGR | From net `V`. CAGR uses `252` days/year. |
| X-02 | Ann. vol, Sharpe | `std(r_net)·√252`; `mean(r_net)/std(r_net)·√252` (rf = 0). |
| X-03 | Sortino | `mean(r_net)·252 / (sqrt(mean(min(r_net,0)²))·√252)`, MAR = 0. |
| X-04 | Max drawdown, Calmar | `max(1 − V/running_peak)`; `CAGR / MaxDD`. |
| X-05 | CVaR 5 % (daily) | Mean of daily returns at or below the 5th percentile (historical). |
| X-06 | Turnover, cost | Mean daily `Σ|Δw|` over risky assets (vs drifted weights); total cost paid as % of initial `V`. |
| X-07 | Rolling window | 63 trading days for rolling Sharpe/Sortino. |
| X-08 | Per-regime metrics | Computed on the daily returns inside each regime label (§10 B-03). |

---

## 12. Benchmark Spec (Task 5.1)

| ID | Requirement |
|---|---|
| K-01 | **One shared simulator.** Benchmarks call the *same* `apply_trades()` / `mark_to_market()` functions (and the same `CostModel`, slippage mode, execution convention, initial portfolio) as the environment. This is what makes "apples-to-apples" structural rather than hoped-for (T-13). |
| K-02 | **Buy-and-Hold:** equal weights at start, never trade → cost 0. Identical to the RL agent's always-HOLD policy (T-08). |
| K-03 | **1/N:** rebalance to 0.2 each every 21 trading days (`rebalance_every`), paying 10 bps on the drift-correcting trades. |
| K-04 | **Mean-Variance (SciPy SLSQP):** maximize Sharpe (rf = 0), long-only, `Σw = 1`, bounds `[0,1]`, initial guess 1/N. Estimation: sample mean and covariance of the **trailing 252 days of simple returns ending at `t`** (+ `1e-6·I` ridge). Rebalance every 21 days. If `res.success` is False, keep previous weights and log it. History from before the test start (2019) may be used (it is the past). Optional: min-variance variant. |
| K-05 | Same metrics (§11), same regimes, same CSV contract (§13). |

---

## 13. Data Contracts & Dashboard (Task 5.2)

| ID | Contract |
|---|---|
| C-01 | `results/backtests/{strategy}.csv` (RL: `{model}_seed{k}.csv`, plus `{model}_mean.csv`). Columns, in order: `date, portfolio_value, daily_return, cost_paid, turnover, w_NVDA, w_AAPL, w_MSFT, w_GOOGL, w_AMZN, w_CASH, regime, drawdown, rolling_sharpe_63, rolling_sortino_63`. `date` = ISO `YYYY-MM-DD`. |
| C-02 | `results/metrics_summary.csv`: `strategy, seed, regime, slippage_mode, total_return, cagr, ann_vol, sharpe, sortino, max_drawdown, calmar, cvar_5, avg_turnover, total_cost`. `seed` = `agg` for cross-seed aggregates. |
| C-03 | `results/manifest.json`: config hash, git commit, creation time, file list. |
| C-04 | **Dashboard is read-only.** It reads only the CSVs above (cached with `@st.cache_data`) and validates the schema on load; a missing or malformed file yields a friendly error, never a stack trace. No `yfinance` calls, no training, no model inference inside the dashboard. |

---

## 14. Tests & Invariants (pytest — must be green before merge)

| ID | Test |
|---|---|
| T-01 | Cleaned prices: no NaN/inf, monotonic unique index, prices > 0. |
| T-02 | Feature causality (truncation test, F-05). |
| T-03 | Scaler fit-window: stats equal those from the train slice only; test path loads from disk and never calls `fit`. |
| T-04 | Action table: 31 entries, bijective, HOLD = 0, no `from == to`. |
| T-05 | 10,000 random-policy steps: weights ≥ 0 and `|Σw − 1| < 1e-9` at every step. |
| T-06 | 100 consecutive HOLD steps with drifting weights ⇒ total cost exactly 0. |
| T-07 | Hand-computed cost: `V=1`, `w=[.2,.2,.2,.2,.2,0]`, move 10 % NVDA→AAPL ⇒ cost = `0.001 × 0.2 × 1.0 = 0.0002`, `V_post = 0.9998`, weights `[.1,.3,.2,.2,.2,0]`. Cash→asset 10 % ⇒ cost `0.0001`. |
| T-08 | Always-HOLD equity curve == Buy-and-Hold equity curve (tolerance 1e-9). |
| T-09 | Reward decomposition equals the RW-01 formula (cost counted once), tolerance 1e-6. |
| T-10 | `check_env` passes; `reset(seed=s)` is reproducible. |
| T-11 | Quantile-regression sanity: single-state bandit with returns ~ N(μ, σ); learned quantiles match the analytic inverse CDF within tolerance. |
| T-12 | CVaR slicing: unsorted input is sorted first; `α=1.0` ⇒ mean; `α=0.2, N=20` ⇒ k = 4. |
| T-13 | Fee parity: a fixed-target rebalance run through the env and through the benchmark simulator gives the same equity curve. |
| T-14 | Metrics module matches hand-computed values on a tiny known series. |
| T-15 | Final-backtest script refuses to run without a matching `config/FROZEN`. |
| T-16 | Slippage: median slip in `shock_*` windows > median slip in `normal` days. |
| T-17 | Dashboard loader: missing / malformed CSV ⇒ handled error message, no crash. |

---

## 15. Known Limitations (must appear in the final report)

| ID | Limitation |
|---|---|
| L-01 | **Hindsight selection bias.** The universe is today's mega-caps picked with knowledge of 2023; NVDA alone rose by a very large multiple over the sample. Absolute returns are inflated for *every* strategy — interpret results **relatively**, not absolutely. |
| L-02 | One historical path, 4-year test, essentially one crash → low statistical power. No significance claims without bootstrap CIs (optional task). |
| L-03 | Training data (2012–2019) contains no COVID-scale crash; stress windows are out-of-distribution. Failure there is a legitimate finding. |
| L-04 | Market-on-close fills, 0 % cash yield, no market impact, no taxes. |
| L-05 | RL variance across seeds is large; single-seed conclusions are invalid. |
| L-06 | The 10 % step action space limits how fast the agent can de-risk (≥ 10 days to reach full cash). |

---

## 16. Config Skeleton (`config/config.yaml`)

```yaml
data:
  tickers: [NVDA, AAPL, MSFT, GOOGL, AMZN]
  start: "2012-01-01"
  end: "2024-01-01"          # exclusive
  auto_adjust: true
  ffill_limit: 5
split:
  train: ["2012-01-01", "2019-12-31"]
  tune_train: ["2012-01-01", "2017-12-31"]
  validation: ["2018-01-01", "2019-12-31"]
  test: ["2020-01-01", "2023-12-31"]
features:
  sma_short: 10
  sma_long: 50
  rsi: 14
  vol: 20
  volume: 20
  clip: 5.0
env:
  initial_value: 1.0
  initial_weights: equal      # 0.2 each, cash 0
  step: 0.10
  cost: 0.0010
  slippage_mode: none         # none | vol_scaled
  slippage_s0_bps: 2
  execution: close            # close | next_open
  include_weights_in_obs: true
  episode_len: 252
reward:
  lambda1: null               # set after tuning (RW-03)
  lambda2: null
  downside_window: 20
  reward_scale: 100
agents:
  gamma: 0.99
  hidden: [128, 128]
  lr: 1.0e-4
  batch: 64
  replay: 100000
  learn_start: 1000
  target_update: 500
  eps: [1.0, 0.05, 0.30]      # start, end, fraction of steps
  train_steps: 200000
  n_quantiles: 20
  kappa: 1.0
  alpha_cvar: 0.2
  seeds: [0, 1, 2, 3, 4]
regimes:
  shock_covid: ["2020-02-19", "2020-03-23"]
  shock_rates: ["2022-01-03", "2022-10-12"]
  high_vol_multiple: 2.0
benchmarks:
  rebalance_every: 21
  mvo_window: 252
metrics:
  periods_per_year: 252
  rolling_window: 63
```

---

## 17. Repository Layout

```
rl-portfolio/
├── specs.md  TODO.md  README.md  requirements.txt  Makefile
├── config/            config.yaml  FROZEN (written only at freeze)
├── data/              raw/  processed/
├── artifacts/         scaler_*.json  action_table.json  models/
├── src/
│   ├── data/          download.py  clean.py
│   ├── features/      indicators.py  scaler.py  build_features.py
│   ├── env/           portfolio.py (shared simulator)  costs.py  reward.py  portfolio_env.py
│   ├── agents/        tabular_q.py  dqn.py  qrdqn.py  replay.py  train.py
│   ├── benchmarks/    buy_hold.py  equal_weight.py  mvo.py
│   └── evaluation/    metrics.py  regimes.py  backtest.py  export.py
├── scripts/           run_final_backtest.py  freeze_config.py
├── dashboard/         app.py  loaders.py
├── results/           backtests/  training/  metrics_summary.csv  manifest.json
├── reports/           data_quality.md  feature_shift.md  stress_test.md  final_report.md
└── tests/             test_*.py (T-01 … T-17)
```

---

## 18. Decision Log & Open Questions

### 18.1 Decision Log (all **Proposed** until the team acknowledges — then mark Accepted)

| ID | Roadmap said | Spec says | Why |
|---|---|---|---|
| DL-01 | 4 indicators, 5 × 6 obs | 6 features: add `ret_1d`, `vol_chg`; SMAs as price gaps | 5×6 needs 6 features; raw SMAs are non-stationary. |
| DL-02 | 30-D observation | 36-D = 30 market + 6 current weights (`include_weights_in_obs: false` reproduces the 30-D version) | Without weights the agent cannot observe its holdings, so costs and drawdown are unobservable (partial observability). |
| DL-03 | Unspecified start | All strategies start equal-weight, `V0 = 1`, entry not charged | Always-HOLD ≡ Buy-and-Hold gives a free correctness test and removes start asymmetry. |
| DL-04 | Adj Close + raw O/H/L | `auto_adjust=True` for all OHLC | Mixing adjusted and unadjusted prices corrupts indicators. |
| DL-05 | "Tune λ empirically" | Grid + validation slice + tune once with DQN + degeneracy guard | Avoids tuning on test; avoids favoring the proposed model; catches the all-cash collapse. |
| DL-06 | Distortion wired into the loss | Full-quantile loss; CVaR at action selection (acting and target) | Training only low quantiles biases the distribution. |
| DL-07 | Unspecified | Fill at close of `t` (MOC); `next_open` as robustness check | Standard convention, explicit and switchable. |
| DL-08 | Slippage mentioned only in Task 4.2 | Separate vol-scaled layer; headline = 10 bps only | Keeps headline numbers equal to the specified cost model. |
| DL-09 | "COVID / war / rate hikes" | Fixed calendar windows (B-03) + data-driven `high_vol` flag | Prevents post-hoc window cherry-picking; war and rates overlap in 2022. |
| DL-10 | End date 2023-12-31 | `end="2024-01-01"` (exclusive) | yfinance `end` is exclusive. |
| DL-11 | "Discretize states" for tabular Q | 135-state compressed state (M-02) | A 36-D continuous state cannot be tabulated. |
| DL-12 | "Never compute rolling metrics across the entire dataset" | Causal rolling features on the continuous series are required; leakage = normalization fit, bfill, centered windows | Computing test features separately yields NaN/incorrect warm-up. |
| DL-13 | No validation set | Tune-train 2012–2017 / validation 2018–2019 | Needed for GR/S-02. |
| DL-14 | Reward unscaled | `reward_scale = 100`; λ grid sized to daily-return scale | Daily returns ≈ 1e-3; unscaled rewards stall learning. |
| DL-15 | Benchmarks pay 10 bps | Benchmarks share the env's simulator code | Structural fairness (K-01). |

### 18.2 Open Questions (answer, then move to the Decision Log)

| ID | Question | Default until answered |
|---|---|---|
| OQ-01 | Should cash earn a T-bill yield? | No (0 %). |
| OQ-02 | Add SPY as a market reference line on the dashboard? | No. |
| OQ-03 | Bootstrap confidence intervals for Sharpe differences? | Optional / stretch. |
| OQ-04 | Is next-open execution required as a robustness check or optional? | Optional. |

---

## 19. Change Process

1. Open a PR that edits `specs.md`, adds a Decision Log row, and bumps the version in the header.
2. Update `TODO.md` (affected tasks, Done-when criteria) in the same PR.
3. At least one other teammate approves. No silent deviations — code that contradicts this file is a bug (GR-1).
4. After `config/FROZEN` exists, spec changes touching §4–§12 invalidate the freeze and all test results; re-run Phase 4.
