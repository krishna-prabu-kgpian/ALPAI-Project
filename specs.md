# specs.md — Risk-Sensitive RL Portfolio Allocation

**Two-stage design (market-regime model → allocator) · curriculum training · QR-DQN with CVaR action selection**

| | |
|---|---|
| **Version** | **1.1** (supersedes 1.0; new decisions DL-16…DL-26 need team acknowledgement) |
| **Date** | 2026-11-06 |
| **Companion file** | `TODO.md` (step-by-step plan; refers to the rule IDs below) |
| **Authority** | This file is the **single source of truth**. If code, a notebook, a chat message or an AI answer disagrees with it, **this file wins** — or you amend it first (§22). |

**What changed vs v1.0 (one-minute summary):**
1. Training now **includes COVID (2020) and 2022–2023** via **curriculum learning** (§10). They are no longer test data.
2. New **test scenarios**: a *normal* year (2024) and an *Iran-war* window (2026-02-27 → 2026-04-30) (§5).
3. New **Stage 1: regime model** that feeds 3 regime probabilities to every allocator (§7).
4. Quick-tuning plan, fewer seeds, and an ordered **time-box** (§12).
5. **Pre-registered success criteria** SC-1…SC-4 so "satisfactory" is defined before we see results (§2).
6. Data now runs to 2026-04-30; rolling metrics use a 21-day window; table formatting bug fixed (no `|` inside cells).

---

## 0. Design at a Glance

```
 prices ─► per-stock features (30) ───────────────┐
        └► market features (3) ─► STAGE 1: regime model ─► P(Calm, Choppy, Crisis)
                                                          │
          observation (39) = [30 stock features | 6 weights | 3 regime probabilities]
                                                          │
 STAGE 2: allocator (Tabular-Q | DQN | QR-DQN) picks 1 of 31 daily trades
          trained with a CURRICULUM:  calm  →  COVID  →  rate-hike stress
                                                          │
 compared with Buy&Hold | 1/N | Mean-Variance  (identical 10 bps fees)
```

**Timeline (who may use which years):**

| Years | Role | Used for |
|---|---|---|
| 2012–2019, 2021 | **calm** window | curriculum stage 1 (and all later stages as "memory") |
| 2020 | **covid** window | curriculum stage 2 |
| 2022–2023 | **rates** window | curriculum stage 3 |
| 2022 | validation (tuning runs only) | tuning runs train on 2012–2021 and are scored on 2022 |
| 2024 | **TEST A — normal** | blind, after freeze |
| 2025-01-01 → 2026-02-26 | excluded | never trained, tuned or scored on (rows kept so rolling features stay continuous) |
| 2026-02-27 → 2026-04-30 | **TEST B — Iran war** | blind, after freeze |

---

## 1. Ground Rules (read before touching code)

| ID | Rule |
|---|---|
| GR-1 | **Spec wins.** Never "fix" a discrepancy silently in code. Either the code is wrong or the spec needs a logged amendment. |
| GR-2 | **Unspecified = unknown.** If this file doesn't define it, don't invent it. Add an Open Question (§21.2) and ask. |
| GR-3 | **No magic numbers.** Every constant lives in `config/config.yaml` (§19) and mirrors §3. |
| GR-4 | **Hypotheses are not results.** H1–H5 stay hypotheses until a committed script produces evidence. Negative results are valid and must be reported. |
| GR-5 | **Verify library APIs against the installed version** (`yfinance`, `gymnasium`, `torch`, `scikit-learn` change often). Run a 3-line probe or read `help()`. |
| GR-6 | **Every reported number is traceable**: committed script + config hash, recorded in `results/manifest.json`. |
| GR-7 | **Test windows are blind** (S-03). |
| GR-8 | **AI-assisted code must pass the tests in §17 before merge.** If an AI suggests changing a constant or formula from this file, treat it as a spec amendment (§22), not a code tweak. |
| GR-9 | **"Done" = Done-when criteria in `TODO.md` met + tests green + `TODO.md` updated in the same commit.** |
| GR-10 | **Pre-registration.** Scenario windows (S-02), phase windows (B-03) and success criteria (SC-1…SC-4) are fixed **before** any test run. Changing them afterwards invalidates the test results (§22). |

---

## 2. Goal, Hypotheses, Success Criteria

**Research question.** Does a two-stage design — a small market-regime model feeding a distributional RL allocator trained with a curriculum — allocate a 5-stock portfolio better, net of 10 bps costs, than a mean-based DQN, a tabular Q-learner and classical benchmarks, especially in a geopolitical shock it never saw in training?

**Hypotheses (to be tested, not assumed):**

| ID | Hypothesis |
|---|---|
| H1 | Standard DQN suffers deeper drawdowns than QR-DQN(α=0.2) in the war window. |
| H2 | QR-DQN(α=0.2) has lower max drawdown and CVaR than QR-DQN(α=1.0) at similar or modestly lower return (isolates CVaR from the reward penalty). |
| H3 | If RL agents beat the benchmarks, it shows in drawdown/tail metrics, not raw return. |
| H4 | Stage 1 helps: regime-ON beats regime-OFF (control A1). *Supported* if, median over seeds, regime-ON has lower war-window max drawdown **and** a Normal-year Sortino that is not lower. |
| H5 | Curriculum helps: curriculum-ON beats curriculum-OFF (control A2), same support rule as H4. |

**Success criteria (pre-registered; "headline model" = the model nominated by TU-04):**

| ID | Criterion | Pass rule |
|---|---|---|
| SC-1 | Tail protection in the war | Headline mean-over-seeds max drawdown in `test_war` is **lower than both** Buy&Hold and 1/N. |
| SC-2 | Tail protection overall | CVaR 5 % on the pooled days of both test scenarios is **not worse than** Buy&Hold and 1/N. |
| SC-3 | Normal-year efficiency | Sortino in `test_normal` is at least **0.8 × the best** of {Buy&Hold, 1/N, MVO}. (If that best Sortino is ≤ 0, the headline must be ≥ it.) |
| SC-4 | No collapse | The degeneracy guard (RW-04) passes in both test scenarios. |

**Verdict:** "Satisfactory" only if SC-1…SC-4 all hold; otherwise "not satisfactory" — still a valid, reportable result. Also report, descriptively, the rebound given up in `war_recovery` (return vs Buy&Hold).

**Non-goals.** No shorting, leverage, derivatives, intraday data, live trading, taxes, or claims of real-world profitability.

---

## 3. Notation & Global Constants

| Symbol | Meaning |
|---|---|
| `A` | Assets, **fixed order everywhere**: `[NVDA, AAPL, MSFT, GOOGL, AMZN]` |
| `B` | Buckets = `A + [CASH]` (6 entries, CASH is index 5) |
| `t` | Trading-day index (close of day `t`) |
| `P_{i,t}` | Adjusted close of asset `i` on day `t` |
| `r_{i,t+1}` | Simple return `P_{i,t+1}/P_{i,t} − 1` (cash: 0) |
| `V_t` | Portfolio value at close of `t`, before trading on `t` |
| `w_t^pre` / `w_t^tgt` | Drifted weights before trading / target weights after the action (6-vectors, each sums to 1) |
| `c` | Proportional transaction cost = 0.0010 (10 bps) |

| Parameter | Value |
|---|---|
| Universe | `NVDA, AAPL, MSFT, GOOGL, AMZN` (daily data) |
| Download window | start `2012-01-01`, end `2026-05-01` (**exclusive** → last row 2026-04-30) |
| Curriculum windows | `calm` = 2012-01-01→2019-12-31 plus 2021-01-01→2021-12-31; `covid` = 2020-01-01→2020-12-31; `rates` = 2022-01-01→2023-12-31 |
| Fit window — final | 2012-01-01 → 2023-12-31 (scaler, regime model, tabular cut-points) |
| Fit window — tuning | 2012-01-01 → 2021-12-31 |
| Validation (tuning only) | 2022-01-01 → 2022-12-31, label `val_2022` |
| Test A (normal) | 2024-01-01 → 2024-12-31, label `test_normal` |
| Test B (war) | 2026-02-27 → 2026-04-30, label `test_war`; phases `war_drawdown` 2026-02-27→2026-03-30, `war_recovery` 2026-03-31→2026-04-30 |
| Excluded | 2025-01-01 → 2026-02-26 |
| Transaction cost `c` | 0.0010 on traded notional of **risky assets** |
| Rebalance step | 0.10 |
| Initial portfolio | `V0 = 1.0`, equal weights 0.2 × 5, cash 0 (all strategies, every episode) |
| Cash yield | 0 % |
| Annualization / risk-free | 252 days / 0 |
| Feature windows | SMA 10, SMA 50, RSI 14, vol 20, volume 20 (market features: 20, 252) |
| z-score clip | ±5 |
| Regimes | K = 3 (Calm, Choppy, Crisis); EMA span 5 |
| Episode length (training) | 126 trading days |
| Discount `γ` | 0.99 |
| Seeds | `0, 1, 2` |
| Training steps | 150,000 (final and validation comparison); 75,000 (λ tuning runs) |

---

## 4. Data Spec

| ID | Requirement |
|---|---|
| D-01 | **Call:** `yfinance.download(tickers, start="2012-01-01", end="2026-05-01", interval="1d", auto_adjust=True, progress=False)`. `end` is exclusive. Expect first row 2012-01-03, last row 2026-04-30. |
| D-02 | **Adjustment consistency.** `auto_adjust=True` gives split- *and* dividend-adjusted O/H/L/C (no separate "Adj Close"). Never mix an adjusted close with unadjusted O/H/L. Internal column `close` = adjusted close. Probe the returned structure first (multi-ticker downloads return MultiIndex columns in current versions). |
| D-03 | **Cache raw data** to `data/raw/prices_raw.parquet` + `prices_raw.meta.json` (UTC timestamp, yfinance version, row counts, SHA-256). Downstream code reads only the cache. Re-download only via `--refresh` **and** a Decision Log entry (Yahoo data can be revised retroactively). |
| D-04 | **Calendar.** All five tickers share one exchange calendar → assert identical date indexes. If not, reindex to the union and apply D-05. |
| D-05 | **Missing data.** `ffill(limit=5)` on prices only. **Never `bfill`, never interpolate** (both use future information). Missing volume → 0. Log filled-cell counts; if more than 0.1 % of cells, stop and investigate. After warm-up, assert zero NaN/inf. |
| D-06 | **Sanity checks:** strictly increasing unique `DatetimeIndex`; prices > 0; volume ≥ 0; row count in 3,580–3,630; print the 10 largest absolute daily returns per asset — any move above 25 % must be explained (e.g. an unadjusted split) before proceeding. *These are integrity checks only; no performance analysis on test windows.* |
| D-07 | **Outputs:** `data/processed/prices_clean.parquet`, `reports/data_quality.md`. |

---

## 5. Split & Scenario Spec

| ID | Requirement |
|---|---|
| S-01 | **Chronology rule.** Every date used for training, tuning, scaling or regime-fitting is earlier than every test date. The *order* in which training windows are shown to the agent is free (that is what a curriculum is); only train-before-test matters. |
| S-02 | **Windows** are exactly those in §3. Scenario labels: `val_2022`, `test_normal`, `test_war`; phases `war_drawdown`, `war_recovery`. |
| S-03 | **Test guard.** Test metrics may only be produced by `scripts/run_final_backtest.py`, which refuses to run unless `config/FROZEN` holds the hash of the current `config.yaml` (T-15). Test windows must also never be used for scaling, regime fitting, curriculum design or hyperparameter choice. Any violation contaminates the test set: log it and disclose it. |
| S-04 | **Tuning split.** Tuning and validation-comparison runs use the **tuning pool** = curriculum windows ∩ 2012–2021 (calm and covid; `rates` absent), fit window 2012–2021, and are scored on `val_2022`. |
| S-05 | **Final training** uses all three windows (2012–2023) with fit window 2012–2023. |
| S-06 | **Why 2025 is excluded:** it contains the April 2025 tariff crash (would contaminate a "normal" test) and training on it would leak into the 2024 test. Its rows stay in the data so rolling features and the regime EMA are continuous into 2026. |
| S-07 | **Warm-up.** Features need up to 50 days (market drawdown feature: expanding window up to 252). Leading NaN rows are dropped (training effectively starts ≈ mid-March 2012). Evaluation windows use the continuous series, so their first day already has full history (see F-01). |
| S-08 | **Tuning cannot see 2022 in training; final models can.** After tuning, 2022 joins the training pool (`rates` window). The only held-out stress is therefore the war window. |

---

## 6. Feature Spec

**F-01 — What "no look-ahead" actually means here.**
- ✅ **Allowed and required:** *causal* (trailing) rolling windows computed on the **continuous** full series, then split by date. Computing evaluation-window features separately would make their first days NaN or wrong.
- ❌ **Look-ahead bugs:** centered windows, `bfill`, `shift(-k)`, normalization or regime statistics fit beyond the fit window, off-by-one date alignment.

**F-02 — Six features per asset (fixed order; stationary by construction).**

| # | Name | Definition |
|---|---|---|
| 0 | `ret_1d` | `log(P_t / P_{t-1})` |
| 1 | `sma10_gap` | `P_t / SMA10_t − 1` |
| 2 | `sma50_gap` | `P_t / SMA50_t − 1` |
| 3 | `rsi14` | Wilder RSI via `ewm(alpha=1/14, adjust=False, min_periods=14)` on gains/losses; avg loss = 0 → 100 (50 if avg gain is 0 too) |
| 4 | `vol20` | `std(ret_1d, 20, ddof=1) × sqrt(252)` |
| 5 | `vol_chg` | `log((Volume_t + 1) / (mean(Volume over last 20 days incl. t) + 1))` |

**F-03 — Returns** always use the adjusted close.

**F-04 — Normalization.** `z = (x − μ_fit) / max(σ_fit, 1e-8)` per (asset, feature) → 30 means and 30 stds, then clip to `[−5, +5]` (fixed bounds).
- `μ_fit, σ_fit` come from the **fit window only**: 2012–2023 for final models, 2012–2021 for tuning runs.
- Persist to `artifacts/scaler_{fit_window}.json` (e.g. `scaler_2012-2023.json`). Evaluation paths **load** the scaler; they never refit.
- Evaluation-window z-scores will not have mean 0 / std 1. That is expected, not a bug.

**F-05 — Causality test (T-02).** For random cut dates `t`, every feature (F-02 **and** F-07) at `t` computed from `data[: t+1]` must equal the value computed from the full data.

**F-06 — Layout.** Columns are a MultiIndex `(asset, feature)`. The flattened per-stock block is **asset-major**: `block[asset_idx*6 + feature_idx]`. Outputs: `data/processed/features_raw.parquet`, `data/processed/features_z_{fit_window}.parquet`.

**F-07 — Three market-level features (inputs to Stage 1; same value for all assets).** Let `basket_ret_t` = mean of the 5 simple daily returns (daily-rebalanced equal-weight basket) and `I_t` = cumulative product of `(1 + basket_ret)` starting at 1.0.

| Name | Definition |
|---|---|
| `m_vol20` | `std(basket_ret, 20, ddof=1) × sqrt(252)` |
| `m_corr20` | Mean of the 10 pairwise Pearson correlations of `ret_1d` over the trailing 20 days (a pair with zero variance counts as 0) |
| `m_dd` | `1 − I_t / max(I over the trailing 252 days incl. t)` (use `min_periods=1`, so early rows use a shorter window) |

Output: `data/processed/market_features_raw.parquet`. These features are **not** placed in the observation directly — only the regime probabilities derived from them are (§7).

---

## 7. Stage 1 — Regime Model

*Purpose: give every allocator the same compact description of "what kind of market is this?" (volatility, cross-stock correlation, drawdown).*

| ID | Requirement |
|---|---|
| RG-01 | **Inputs:** the 3 features of F-07, z-scored with a **regime scaler** (same `Scaler` logic as F-04, clip ±5) fit on the fit window. |
| RG-02 | **Model:** `sklearn.mixture.GaussianMixture(n_components=3, covariance_type="diag", n_init=5, random_state=0)` fit on the fit window only. Deliberately tiny: crisis days are few, so a bigger model would fit noise. |
| RG-03 | **Label order is fixed:** sort components by their mean `m_vol20` (in z units), ascending → `0 = Calm`, `1 = Choppy`, `2 = Crisis`. (Cluster numbers can swap between refits; this ordering prevents it.) |
| RG-04 | **Causal posterior:** the probability vector for day `t` is `predict_proba` of **day-`t` features only**. No HMM smoothing, no sequence model — smoothed estimates would use future days. |
| RG-05 | **EMA smoothing (anti flip-flop):** `p̃_t = a·p_t + (1−a)·p̃_{t−1}`, `a = 2/(span+1)` with `span = 5`; `p̃_first = p_first`; renormalize rows to sum to 1. Precompute over the **continuous** series (no reset per episode) → `data/processed/regime_probs_{fit_window}.parquet`. |
| RG-06 | **Fit windows:** final 2012–2023, tuning 2012–2021, **never** any test window. Persist `artifacts/regime_{fit_window}.joblib` (scaler + GMM + label order). Validation (`val_2022`) uses the **tuning** regime model. |
| RG-07 | **Optional novelty score** (config `regime.include_novelty`, default **false**): `novelty_t = clip((nll_t − p50) / (p99 − p50), 0, 2)` with `nll_t = −score_samples(x_t)` and `p50, p99` the 50th/99th percentiles over the fit window. Appended to the observation when on. |
| RG-08 | **Usage:** (a) appended to the observation (E-04); (b) tabular state (M-02); (c) dashboard timeline (C-01). `regime.enabled: false` removes it everywhere (control A1). |
| RG-09 | **Reports:** `reports/regime_report.md` before the freeze covers only the fit/validation windows (share of days per regime in each curriculum window; mean features per regime). How **test** days map to regimes is reported only **after** the freeze (B-07). |
| RG-10 | **Known weakness (do not hide):** a shock of a new type (the 2026 war was an oil/inflation shock, unlike COVID's volatility spike) may be mapped to the nearest familiar regime, with unreliable probabilities. Control A1 and the optional novelty score exist to expose this. |

---

## 8. Environment Spec

| ID | Requirement |
|---|---|
| E-01 | **Gymnasium API (not legacy `gym`).** `reset(seed=None, options=None) -> (obs, info)`; `step(a) -> (obs, reward, terminated, truncated, info)` (5 values). Must pass `gymnasium.utils.env_checker.check_env` (T-10). |
| E-02 | **Time & execution convention** — the section most likely to be implemented wrongly; see the 7-step order below. |
| E-03 | **Costs apply only to trades**, never to passive drift (steps 3–4 below). |
| E-04 | **Observation:** `Box(shape=(D,), float32)`, default `D = 39`, concatenated in this order: `[0:30]` z-scored per-stock features (F-06 order); `[30:36]` current drifted weights `w_t^pre` as `[NVDA, AAPL, MSFT, GOOGL, AMZN, CASH]`; `[36:39]` smoothed regime probabilities `[Calm, Choppy, Crisis]` (RG-05). Flags: `include_weights_in_obs`, `regime.enabled` (off → `D = 36`, control A1), `regime.include_novelty` (on → one extra value at the end). |
| E-05 | **Action space:** `Discrete(31)`. `0` = HOLD. `1..30` = ordered pairs `(from, to)`, `from ≠ to`, over the 6 buckets, enumerated row-major. Decode table built once, saved to `artifacts/action_table.json`, bijection-tested (T-04). |
| E-06 | **Action semantics:** move `min(0.10, w_pre[from])` of portfolio value from bucket `from` to bucket `to`. If `w_pre[from] < 1e-9` the action is a no-op (cost 0). Long-only, no leverage. |
| E-07 | **Invariants after every step:** all weights ≥ 0 and `abs(sum(w) − 1) < 1e-9`. Assert in debug mode (T-05). |
| E-08 | **Episodes.** *Training:* the curriculum sampler (CU-04) supplies `options={"start_idx": ..., "length": 126}` to `reset`. *Evaluation:* `options={"window": "val_2022" or "test_normal" or "test_war"}` gives one deterministic pass over that window with ε = 0. Every episode starts from the initial portfolio (equal weights). `truncated=True` at the last day that still has a valid `t+1`; `terminated` is never True (no leverage → no ruin). The running peak (drawdown) and the downside window reset at each episode start. |
| E-09 | **Info dict** per step: `date, V, cost, turnover, weights (6), regime_probs (3), r_gross, r_net, drawdown, sigma_down, action_id`. The backtest logger consumes this — nothing is recomputed elsewhere. |

**E-02 / E-03 — Order of operations inside `step(a_t)`:**

1. The agent observes `obs_t` (features through the close of day `t`, `w_t^pre`, regime probabilities of day `t`).
2. Decode `a_t` → target weights `w_t^tgt` (E-06).
3. **Traded notional** per risky asset: `abs(w_t^tgt,i − w_t^pre,i) × V_t`, measured against the **drifted** weights. So a HOLD costs exactly zero even though weights drift passively.
4. **Cost** = `c × (sum of traded notional over the 5 stocks)`; CASH is not charged. Moving 10 % from stock A to stock B pays on both legs (0.2 × c × V); cash↔stock pays one leg. Optional slippage (B-05) is added here.
5. `V_post = V_t − cost`. Costs are deducted proportionally from all holdings, so post-trade weights equal `w_t^tgt` exactly (E-07 preserved).
6. Hold to the close of `t+1`: each holding grows by `(1 + r_{i,t+1})`, cash by 1. `V_{t+1}` = sum of holdings; `w_{t+1}^pre` = holdings / `V_{t+1}`.
7. Compute the reward (§9) and return `obs_{t+1}`.

**A-01 (DL-07).** Trades fill at the close of day `t`, the same close that fed the features (market-on-close). Optional robustness check: `execution: next_open`. Headline results use `close`.
**A-02.** Cash earns 0 %.

---

## 9. Reward Spec

**RW-01 — Formula (cost counted exactly once):**

```
reward_t = reward_scale × ( r_gross − cost_t / V_t − λ1 · σ_down − λ2 · DD )
```

- `r_gross = Σ_i w_t^tgt,i · r_{i,t+1}` — return of the post-trade portfolio **before** cost.
- `cost_t / V_t` — the cost fraction from step 4. *The roadmap formula `R_portfolio − … − Cost` double-counts cost if `R_portfolio` is already net.* Here `r_gross` is explicitly gross. Portfolio value `V` and all metrics use the net path; the difference to the reward is second-order (T-09 tolerance 1e-6).
- `σ_down` — trailing downside semi-deviation of **net daily portfolio returns in the current episode**: `sqrt(mean(min(r_net, 0)²))` over the last ≤ 20 returns including the current one, target 0, daily units. Fewer than 2 returns → 0.
- `DD = (peak_V − V_{t+1}) / peak_V ≥ 0`, with `peak_V` the running maximum within the episode.
- `reward_scale = 100`. The λ's are defined **before** scaling.

| ID | Requirement |
|---|---|
| RW-02 | **Scale warning.** Daily mean returns are ≈ 5e-4, daily downside deviation ≈ 7e-3. A λ1 of 1 makes holding stocks a guaranteed negative reward → the agent sits in cash. Hence the tiny grid in TU-02. |
| RW-03 | **λ1, λ2 are chosen once** by TU-03 and then frozen for **all** RL agents. |
| RW-04 | **Degeneracy guard.** A run **fails** if, in the scored scenario: mean cash weight > 0.80, **or** HOLD is chosen in > 95 % of steps, **or** action entropy ≈ 0 (never trades), **or** mean daily turnover > 0.5 (churn). Log these four numbers for every run. |
| RW-05 | **Sanity identity.** An always-HOLD policy reproduces Buy&Hold exactly (T-08). |

---

## 10. Curriculum Spec

*Curriculum learning here changes **what the agent trains on and in what proportion over time**. It does not change the reward or the loss.*

| ID | Requirement |
|---|---|
| CU-01 | **Windows** (S-02 / §3): `calm`, `covid`, `rates`. |
| CU-02 | **Fixed stage order: calm → covid → rates** (easy → hard). Step 12 computes a difficulty table (equal-weight basket max drawdown per window) for the report. If the data ranks the windows differently, raise an Open Question — **do not reorder silently**. |
| CU-03 | **Stage schedule** (shares of total training steps; mixes are the sampling probabilities over windows): see table below. |
| CU-04 | **Episode sampling.** At each reset: (1) the global step counter picks the stage; (2) pick a window by that stage's mix; (3) pick a start uniformly among valid starts in that window so the whole 126-day episode lies inside one contiguous segment. Stage changes take effect at the next reset. |
| CU-05 | **Memory.** The replay buffer is **never cleared** across stages. ε resets at the start of stages 2 and 3 (table below) and decays linearly to 0.05 over the first half of that stage. The later stages keep a share of `calm` so the agent does not forget normal markets. |
| CU-06 | **Objective (formal).** The RL objective is unchanged (risk-sensitive reward RW-01; TD Huber loss, or quantile Huber loss for QR-DQN). The curriculum changes only the training distribution: at stage `k` minimize `E over w ~ π_k, episode ~ w of [ loss(θ) ]`, where `π_k` is the stage-`k` mix. |
| CU-07 | **Fairness.** Every RL agent uses the same curriculum. Control **A2** turns it off: all steps sample uniformly over all valid start days of the pool, with the single-stage ε schedule (1.0 → 0.05 over the first 30 % of steps). |
| CU-08 | **Tuning pool.** In tuning/validation runs the `rates` window is absent (it is 2022–2023); stage 3 uses the replacement mix below. |
| CU-09 | **Known risk:** each stress window is a **single historical path**, so the agent can memorize it. Episodes are short (126 days) and sampled randomly inside windows to soften this; it is still disclosed as limitation L-04. |

| Stage | Share of steps | Mix: calm / covid / rates | ε at stage start |
|---|---|---|---|
| 1 | 40 % | 100 / 0 / 0 | 1.0 |
| 2 | 30 % | 40 / 60 / 0 | 0.2 |
| 3 (final) | 30 % | 30 / 30 / 40 | 0.2 |
| 3 (tuning replacement, CU-08) | 30 % | 30 / 70 / 0 | 0.2 |

---

## 11. Model Spec

**M-01 — Shared training defaults** (starting points; changes only via validation + Decision Log):

| Parameter | Value |
|---|---|
| Network | MLP, input `D` (E-04), hidden `[128, 128]`, ReLU, float32, CPU by default |
| Optimizer | Adam, lr `1e-4`, grad-clip norm 10 |
| Batch / replay | 64 / 100,000 (uniform); learning starts after 1,000 steps |
| Target network | hard update every 500 gradient steps |
| Exploration | CU-05 schedule (curriculum on) or CU-07 schedule (off); evaluation uses ε = 0 |
| Training length | 150,000 env steps fixed (no early stopping, no per-model tuning) |
| Seeds | 0, 1, 2 for every model; report all |

**M-02 — Tabular Q-learning.** A 39-D continuous observation cannot be tabulated. Compressed state, 3 × 3 × 5 × 3 = **135 states**:
1. `market_regime` (3): argmax of the smoothed regime probabilities (RG-05).
2. `trend_regime` (3): mean `sma50_gap` across assets, cut at terciles of the fit window.
3. `leader` (5): index of the asset with the highest `sma50_gap`.
4. `cash_bucket` (3): `w_cash` in `<0.05`, `0.05–0.5`, `≥0.5`.

Cut-points come from the fit window only. Q-table zero-initialised, `α = 0.1`, same 31 actions, `γ` and ε schedule. The state function is pure, deterministic and unit-tested.

**M-03 — Standard DQN.** `Q(s, ·)` → 31 outputs. Huber loss (κ = 1) on `r + γ(1 − done) · max_a' Q_target(s', a')`. Plain DQN (not Double DQN); `double_dqn: false` flag allowed.

**M-04 — QR-DQN.**
- Output head `31 × N`, `N = 20`. Quantile midpoints `τ̂_i = (2i − 1) / (2N)`, `i = 1..N` (0.025, 0.075, …, 0.975).
- **The loss trains ALL N quantiles:** `u_ij = r + γ·θ_j^target(s', a*) − θ_i(s, a)`; quantile Huber `ρ(u) = abs(τ̂_i − 1{u<0}) · L_κ(u)`, with `L_κ(u) = ½u²` if `abs(u) ≤ κ`, else `κ(abs(u) − ½κ)`, `κ = 1`. Sum over `i`, mean over `j`, mean over batch. Assert tensor shape `[B, N, N]`.
- **Risk distortion lives in action selection, not in the loss (DL-06).** Training only the low quantiles would bias the learned distribution. Instead: `Q_α(s, a)` = mean of the `k` lowest quantiles, `k = max(1, ceil(round(α·N, 8)))` → for `α = 0.2, N = 20`: **k = 4**. `α = 1.0` ⇒ plain mean (risk-neutral).
- `Q_α` is used for **both** acting **and** choosing `a*` for the TD target.
- **Always sort the N outputs ascending before slicing** (quantile crossing is common; unsorted slicing averages the wrong quantiles).

**M-05 — Experiment matrix** (strategy names are used in every file, table and chart):

| Strategy name | What it is | Regime in obs | Curriculum | Role |
|---|---|---|---|---|
| `tabular_q` | Tabular Q-learning | yes (state) | yes | main |
| `dqn` | Standard DQN | yes | yes | main |
| `qrdqn_a1.0` | QR-DQN, α = 1.0 (risk-neutral) | yes | yes | main (isolates the CVaR effect) |
| `qrdqn_a0.2` | QR-DQN, α = 0.2 | yes | yes | main; **headline candidate** |
| `qrdqn_a0.2_noregime` | same, one-stage | **no** | yes | control A1 (tests H4) — *droppable, TU-05* |
| `qrdqn_a0.2_nocurr` | same, no curriculum | yes | **no** | control A2 (tests H5) — *droppable, TU-05* |
| `buy_hold`, `equal_weight`, `mvo` | benchmarks (§15) | — | — | benchmark |

**M-06 — Artifacts.** `artifacts/models/{strategy}_seed{k}.pt`; `results/training/{strategy}_seed{k}.csv` (reward, loss, validation metrics, the four RW-04 numbers); the config hash is stored in each checkpoint.

**M-07 — Reproducibility.** Seed `numpy`, `torch`, `random` and `env.reset(seed=...)`. Evaluation is deterministic.

---

## 12. Tuning & Time-Box Spec

| ID | Requirement |
|---|---|
| TU-01 | **Only λ1 and λ2 are tuned.** Everything else stays at the defaults in M-01. α ∈ {1.0, 0.2} are model *variants*, not tuning knobs. |
| TU-02 | **λ grid (3 cells):** `C0 = (0, 0)` control, `C1 = (0.05, 0.002)` **default**, `C2 = (0.10, 0.005)`. |
| TU-03 | **Protocol.** Train **DQN** (regime ON, curriculum ON, tuning pool, fit window 2012–2021), 2 seeds per cell, 75,000 steps. Score each by a greedy pass on `val_2022`. Pick the highest median Sortino among cells that pass the RW-04 guard. Write λ1, λ2 to `config.yaml` and the Decision Log. With no time left, use `C1` and label it **"untuned default"** in the report (it is a reasoned guess, not a tested value). Tune with DQN, not QR-DQN, so the reward is not tuned to favor the proposed model. |
| TU-04 | **Headline-model selection rule** (the "how do we choose the model" answer). Train the four main allocators × 2 seeds on the tuning pool (150,000 steps), score on `val_2022`, then: (1) drop runs failing RW-04; (2) rank by **median-over-seeds** Sortino; (3) if the top two are within 0.1 Sortino, prefer lower median max drawdown; (4) if still tied, prefer the simpler model in the order `tabular_q < dqn < qrdqn_a1.0 < qrdqn_a0.2`. Never pick a best seed. Controls A1/A2 are not eligible. The nominated model is the **headline** for SC-1…SC-3; **all** models are still reported on the test scenarios. |
| TU-05 | **Time-box.** Step 15 measures wall-time per 10,000 steps. If the plan does not fit the deadline, cut in this order and log each cut: (1) skip the λ grid, use `C1`; (2) validation comparison with 1 seed; (3) final seeds 3 → 2; (4) steps 150,000 → 100,000; (5) drop control A2; (6) drop control A1 (then H4/H5 are reported as "untested"). |
| TU-06 | **Never cut:** the test-blind rule, the degeneracy guard, tests T-01…T-14 and T-18…T-20, the causality checks. |

---

## 13. Backtest, Scenarios & Stress Spec

| ID | Requirement |
|---|---|
| B-01 | **Final evaluation** runs only after `config/FROZEN` exists (S-03). Each strategy × each seed makes one greedy pass over each scenario (`test_normal`, `test_war`) from the initial equal-weight portfolio, slippage off. Report mean ± std across seeds **and** every individual seed. **Never select a best seed using test results.** |
| B-02 | No retuning and no online learning on test windows. Network weights are frozen. |
| B-03 | **Scenarios and phases are pre-registered (GR-10):** `test_normal`, `test_war`, `war_drawdown` (2026-02-27 → 2026-03-30), `war_recovery` (2026-03-31 → 2026-04-30). The phase boundary uses the S&P 500 trough date from news; Step 24 also reports the basket's own trough date, but the pre-registered split **does not change**. `test_normal` is "normal" only by label — it contains minor events (e.g. the April 2024 Iran–Israel exchange, the August 2024 volatility spike) — so a data-driven day flag `high_vol` (basket `vol20` > 2 × its fit-window median) is reported alongside. |
| B-04 | "Correlations spike, volatility jumps" are **hypotheses to measure**. Step 24 reports the observed mean pairwise 20-day correlation and the basket-vol ratio vs the fit-window median, per scenario and phase. |
| B-05 | **Slippage layer (sensitivity only).** `slippage_mode: vol_scaled`: `slip_bps_{i,t} = s0 × clip(vol20_{i,t} / median_fit(vol20_i), 1, 5)`, `s0 = 2 bps`, charged on traded notional, computed causally, applied **identically to every strategy**. Headline tables: slippage **off**. Sensitivity tables: both modes. |
| B-06 | **Defensive-behaviour analysis (war).** Agents have no foresight, so "de-risked in time" is *measured*, never assumed: mean cash weight at trading day 5, 10, 20 and at the end of the window; max drawdown in `test_war`, `war_drawdown` and `war_recovery`; return given up in `war_recovery` vs Buy&Hold and 1/N. If the agent does not de-risk, say so (GR-4). |
| B-07 | **Post-freeze analysis rule.** How test days map to regimes, novelty, feature-shift statistics for test windows, correlation/vol ratios — all are computed **only after Step 23 finishes**, are descriptive, and cannot change anything upstream. If something upstream *must* change, re-freeze, re-run Step 23 and disclose it. |
| B-08 | **Pooled metrics (SC-2)** concatenate the daily returns of `test_normal` and `test_war` (not a continuous equity path). |

---

## 14. Metrics Spec

*Single implementation in `src/evaluation/metrics.py`, imported by the exporter **and** the dashboard. Never write a second Sharpe formula.*

| ID | Metric | Definition |
|---|---|---|
| X-01 | Total return, CAGR | From net `V`. CAGR uses 252 days per year and is left **blank for windows shorter than 252 days** (e.g. `test_war`); use total return there. |
| X-02 | Annualized vol, Sharpe | `std(r_net)·√252`; `mean(r_net)/std(r_net)·√252` (risk-free = 0). |
| X-03 | Sortino | `mean(r_net)·252 / (sqrt(mean(min(r_net,0)²))·√252)`, target 0. |
| X-04 | Max drawdown, Calmar | `max(1 − V/running_peak)`; `CAGR / MaxDD` (blank when CAGR is blank). |
| X-05 | CVaR 5 % (daily) | Mean of the daily returns at or below the 5th percentile (historical). |
| X-06 | Turnover, total cost | Mean daily sum of `abs(Δw)` over the 5 stocks (vs drifted weights); total cost paid as % of initial `V`. |
| X-07 | Rolling window | **21 trading days** for rolling Sharpe/Sortino (the war window is only ≈ 44 days); first 20 rows are blank. |
| X-08 | Per-scenario / per-phase | Computed on the daily returns inside each scenario or phase (B-03); pooled variant per B-08. |

---

## 15. Benchmark Spec

| ID | Requirement |
|---|---|
| K-01 | **One shared simulator.** Benchmarks call the *same* `apply_trades()` / `mark_to_market()` and the same `CostModel`, slippage mode, execution convention and initial portfolio as the environment. That makes "apples-to-apples" structural, not hoped-for (T-13). |
| K-02 | **Buy&Hold:** equal weights at the start of each scenario, never trade → cost 0. Identical to the RL agent's always-HOLD policy (T-08). |
| K-03 | **1/N:** rebalance to 0.2 each every 21 trading days, paying 10 bps on the drift-correcting trades. |
| K-04 | **Mean-Variance (SciPy SLSQP):** maximize Sharpe (risk-free 0), long-only, weights sum to 1, bounds `[0,1]`, initial guess 1/N. Estimate from the **trailing 252 days of simple returns ending at `t`** (+ `1e-6·I` ridge); rebalance every 21 days. If `res.success` is False, keep previous weights and log it. Pre-window history (2023 for Test A, 2025 for Test B) may be used — it is the past. |
| K-05 | Same metrics (§14), same scenarios and phases, same CSV contract (§16). |

---

## 16. Data Contracts & Dashboard

| ID | Contract |
|---|---|
| C-01 | **Backtest CSVs:** `results/backtests/{scenario}/{strategy}.csv` for benchmarks and `{strategy}_seed{k}.csv` for RL agents, plus `{strategy}_mean.csv` (cross-seed mean). `scenario` is `test_normal` or `test_war`. Validation runs: `results/validation/val_2022/…`. Slippage runs: `results/backtests_slippage/{scenario}/…`. **18 columns, in this order:** `date, portfolio_value, daily_return, cost_paid, turnover, w_NVDA, w_AAPL, w_MSFT, w_GOOGL, w_AMZN, w_CASH, p_calm, p_choppy, p_crisis, regime, drawdown, rolling_sharpe_21, rolling_sortino_21`. The first 14 are written by the backtest runner, the last 4 by the exporter. `regime` is one of `val_2022`, `normal`, `war_drawdown`, `war_recovery`. `date` is ISO `YYYY-MM-DD`. |
| C-02 | **`results/metrics_summary.csv`:** `strategy, seed, scenario, phase, slippage_mode, total_return, cagr, ann_vol, sharpe, sortino, max_drawdown, calmar, cvar_5, avg_turnover, total_cost`. `seed` is `agg` for cross-seed aggregates; `scenario` is `test_normal`, `test_war` or `pooled`; `phase` is `all`, `war_drawdown` or `war_recovery`. |
| C-03 | **`results/manifest.json`:** config hash, git commit, creation time, file list. |
| C-04 | **Dashboard is read-only.** It reads only the files above (cached with `@st.cache_data`) and validates the schema on load; a missing or malformed file shows a friendly error, never a stack trace. No `yfinance`, no training, no model inference in the dashboard. |
| C-05 | **`results/verdict.json`:** pass/fail and the numbers behind SC-1…SC-4, the nominated headline model (TU-04), and the H1–H5 support flags. The dashboard shows it on the tear-sheet page. |

---

## 17. Tests & Invariants (pytest — green before merge)

| ID | Test |
|---|---|
| T-01 | Cleaned prices: no NaN/inf, increasing unique dates, prices > 0. |
| T-02 | Feature causality (truncation test, F-05) for per-stock **and** market-level features. |
| T-03 | Scaler fit-window: parameters equal those from the fit slice only; evaluation paths load from disk and never call `fit`. |
| T-04 | Action table: 31 entries, bijective, HOLD = 0, no `from == to`. |
| T-05 | 10,000 random-policy steps: weights ≥ 0 and `abs(sum(w) − 1) < 1e-9` at every step. |
| T-06 | 100 consecutive HOLD steps with drifting weights ⇒ total cost exactly 0. |
| T-07 | Hand-computed cost: `V=1`, `w=[.2,.2,.2,.2,.2,0]`, move 10 % NVDA→AAPL ⇒ cost `0.001 × 0.2 × 1.0 = 0.0002`, `V_post = 0.9998`, weights `[.1,.3,.2,.2,.2,0]`. Cash→stock 10 % ⇒ cost `0.0001`. |
| T-08 | Always-HOLD equity curve equals the Buy&Hold curve (tolerance 1e-9). |
| T-09 | Reward decomposition equals RW-01 (cost counted once), tolerance 1e-6. |
| T-10 | `check_env` passes; `reset(seed=s)` is reproducible. |
| T-11 | Quantile-regression sanity: single-state bandit with returns ~ N(μ, σ); learned quantiles match the analytic inverse CDF within tolerance. |
| T-12 | CVaR slicing: unsorted input is sorted first; `α=1.0` ⇒ mean; `α=0.2, N=20` ⇒ k = 4. |
| T-13 | Fee parity: a fixed-target rebalance run through the env and through the benchmark simulator gives the same equity curve. |
| T-14 | Metrics module matches hand-computed values on a tiny known series (including blank CAGR for short windows). |
| T-15 | The final-backtest script refuses to run without a matching `config/FROZEN`. |
| T-16 | Slippage **formula** (synthetic inputs): equals `s0` when `vol20 ≤ median`, increases monotonically with `vol20`, capped at `5·s0`. (A data-dependent assertion would fail for reasons that are not bugs.) |
| T-17 | Dashboard loader: missing or malformed CSV ⇒ handled error message, no crash. |
| T-18 | Regime model: label order Calm < Choppy < Crisis by mean `m_vol20`; probabilities sum to 1; probabilities at `t` equal when computed from `data[: t+1]` (causal); EMA is causal. |
| T-19 | Curriculum sampler: stage boundaries at 40 % / 70 % of steps; over many draws the window mix matches CU-03 within tolerance; **never** samples a date in 2024 or later; in tuning mode never samples 2022. |
| T-20 | Split integrity: max training/fit date < min test date; tuning pool excludes `val_2022`; excluded 2025 window never appears in any training, tuning, fit or scored window. |

---

## 18. Known Limitations (must appear in the final report)

| ID | Limitation |
|---|---|
| L-01 | **Hindsight selection bias.** The universe is today's mega-caps, picked with knowledge of 2023+; NVDA alone rose by a very large multiple. Absolute returns are inflated for *every* strategy — interpret **relatively**. |
| L-02 | **Low statistical power.** `test_war` is ≈ 44 trading days (one episode), validation is one year, only 3 seeds. No significance claims without bootstrap CIs (optional task). |
| L-03 | **War facts come from news articles.** The S&P trough date may differ from our 5-stock basket's; the conflict's later course after 2026-04-30 is not studied. |
| L-04 | **Single path per stress window** (COVID, 2022–23): the agent can memorize them (CU-09). |
| L-05 | **Stage 1 is tiny and may mis-map a new kind of shock** (RG-10); few crisis days exist in training. |
| L-06 | **`test_normal` is not event-free** (see B-03); `high_vol` days are reported. |
| L-07 | Market-on-close fills, 0 % cash yield, no market impact, no taxes. |
| L-08 | RL variance across seeds is large; single-seed conclusions are invalid. |
| L-09 | **The 10 % step limits how fast the agent can de-risk** (≥ 10 trading days to reach full cash) — significant inside a 44-day window. |
| L-10 | **Tuning is minimal** (3 λ cells, 1 validation year); the headline model is chosen on one year. |
| L-11 | A defensive agent will likely **miss part of the `war_recovery` rebound**; this is reported, not hidden. |

---

## 19. Config Skeleton (`config/config.yaml`)

```yaml
spec_version: "1.1"
data:
  tickers: [NVDA, AAPL, MSFT, GOOGL, AMZN]
  start: "2012-01-01"
  end: "2026-05-01"             # exclusive
  auto_adjust: true
  ffill_limit: 5
windows:                         # inclusive date ranges
  calm:  [["2012-01-01", "2019-12-31"], ["2021-01-01", "2021-12-31"]]
  covid: [["2020-01-01", "2020-12-31"]]
  rates: [["2022-01-01", "2023-12-31"]]
  val_2022: ["2022-01-01", "2022-12-31"]
  test_normal: ["2024-01-01", "2024-12-31"]
  test_war: ["2026-02-27", "2026-04-30"]
  war_drawdown: ["2026-02-27", "2026-03-30"]
  war_recovery: ["2026-03-31", "2026-04-30"]
  excluded: ["2025-01-01", "2026-02-26"]
fit_windows:
  final: ["2012-01-01", "2023-12-31"]
  tuning: ["2012-01-01", "2021-12-31"]
features:
  sma_short: 10
  sma_long: 50
  rsi: 14
  vol: 20
  volume: 20
  market_corr: 20
  market_dd: 252
  clip: 5.0
regime:
  enabled: true                  # false = one-stage control A1
  k: 3
  covariance: diag
  n_init: 5
  random_state: 0
  ema_span: 5
  include_novelty: false
env:
  initial_value: 1.0
  initial_weights: equal         # 0.2 each, cash 0
  step: 0.10
  cost: 0.0010
  slippage_mode: none            # none | vol_scaled
  slippage_s0_bps: 2
  execution: close               # close | next_open
  include_weights_in_obs: true
  episode_len: 126
reward:
  lambda1: null                  # set by TU-03
  lambda2: null
  downside_window: 20
  reward_scale: 100
curriculum:
  enabled: true                  # false = control A2
  stage_share: [0.4, 0.3, 0.3]
  stage_mix: [[1.0, 0.0, 0.0], [0.4, 0.6, 0.0], [0.3, 0.3, 0.4]]   # calm, covid, rates
  tuning_stage3_mix: [0.3, 0.7, 0.0]
  eps_stage_start: [1.0, 0.2, 0.2]
  eps_end: 0.05
  eps_decay_frac_of_stage: 0.5
agents:
  gamma: 0.99
  hidden: [128, 128]
  lr: 1.0e-4
  batch: 64
  replay: 100000
  learn_start: 1000
  target_update: 500
  train_steps: 150000
  tuning_steps: 75000
  n_quantiles: 20
  kappa: 1.0
  alpha_cvar: 0.2
  seeds: [0, 1, 2]
tuning:
  lambda_grid: [[0.0, 0.0], [0.05, 0.002], [0.10, 0.005]]
  default_cell: 1
  seeds: [0, 1]
scenarios:
  high_vol_multiple: 2.0
benchmarks:
  rebalance_every: 21
  mvo_window: 252
metrics:
  periods_per_year: 252
  rolling_window: 21
success:
  sortino_fraction_of_best: 0.8
```

---

## 20. Repository Layout

```
rl-portfolio/
├── specs.md  TODO.md  README.md  requirements.txt  Makefile
├── config/            config.yaml  FROZEN (written only at freeze)
├── data/              raw/  processed/
├── artifacts/         scaler_*.json  regime_*.joblib  action_table.json  tabular_cuts.json  models/
├── src/
│   ├── data/          download.py  clean.py
│   ├── features/      indicators.py  market_features.py  scaler.py  build_features.py
│   ├── regime/        regime_model.py  build_regime.py
│   ├── env/           portfolio.py (shared simulator)  costs.py  reward.py  portfolio_env.py
│   ├── agents/        replay.py  curriculum.py  tabular_q.py  dqn.py  qrdqn.py  train.py
│   ├── benchmarks/    buy_hold.py  equal_weight.py  mvo.py
│   └── evaluation/    metrics.py  scenarios.py  backtest.py  diagnostics.py  export.py
├── scripts/           run_final_backtest.py  freeze_config.py
├── dashboard/         app.py  loaders.py
├── results/           validation/  backtests/{test_normal,test_war}/  backtests_slippage/
│                      training/  tuning/  metrics_summary.csv  verdict.json  manifest.json
├── reports/           data_quality.md  regime_report.md  time_budget.md  stress_test.md  final_report.md
└── tests/             test_*.py (T-01 … T-20)
```

---

## 21. Decisions, Open Questions, Changelog

### 21.1 Decision Log

*Status "Proposed" = needs team acknowledgement (TODO Step 01). "Directed" = set by the project lead.*

| ID | Original roadmap / v1.0 | Spec v1.1 says | Why | Status |
|---|---|---|---|---|
| DL-01 | 4 indicators, 5 × 6 obs | 6 features per stock; SMAs as price gaps | 5×6 needs 6; raw SMAs are non-stationary | Proposed |
| DL-02 | 30-D observation | 36-D (30 + 6 weights) + 3 regime probabilities = **39-D** | Agent must see its holdings; regime is Stage 1's output | Proposed |
| DL-03 | Start unspecified | All strategies start equal-weight, `V0 = 1`, entry not charged | Always-HOLD ≡ Buy&Hold is a free correctness test | Proposed |
| DL-04 | Adj Close + raw O/H/L | `auto_adjust=True` for all OHLC | Mixing adjusted and raw prices corrupts indicators | Proposed |
| DL-05 | "Tune λ empirically" | 3-cell grid, DQN, validation 2022, degeneracy guard, default `C1` | Fast, fair, catches the all-cash collapse | Proposed |
| DL-06 | Distortion wired into the loss | Full-quantile loss; CVaR at action selection (acting and target) | Training only low quantiles biases the distribution | Proposed |
| DL-07 | Execution unspecified | Fill at close of `t` (MOC); `next_open` as optional check | Standard, explicit, switchable | Proposed |
| DL-08 | Slippage only in Task 4.2 | Separate vol-scaled sensitivity layer; headline = 10 bps only | Headline equals the specified cost model | Proposed |
| DL-09 | COVID / rates as test regimes | **Superseded** by DL-16: they are now curriculum training windows | User objective | Superseded |
| DL-10 | End date 2023-12-31 | `end="2026-05-01"` (exclusive) | Covers the war window; yfinance `end` is exclusive | Proposed |
| DL-11 | "Discretize states" | 135-state compressed state; `market_regime` replaces vol bucket | A 39-D state cannot be tabulated | Proposed |
| DL-12 | "No rolling metrics across the whole dataset" | Causal rolling features on the continuous series are required; leakage = fit beyond fit window, bfill, centered windows | Separate evaluation features would be NaN/wrong | Proposed |
| DL-13 | No validation set | Tuning runs train on 2012–2021, validate on 2022 | Needed so test stays blind | Proposed |
| DL-14 | Reward unscaled | `reward_scale = 100`; tiny λ grid | Daily returns are ≈ 1e-3 | Proposed |
| DL-15 | Benchmarks "pay 10 bps" | Benchmarks share the env's simulator code | Structural fairness | Proposed |
| **DL-16** | Train 2012–2019, test 2020–2023 | Train 2012–2023 via curriculum; test = **normal 2024** + **Iran-war window 2026** | Project objective: stress periods are learned, tests are unseen normal + war | **Directed** |
| **DL-17** | Single-stage learner | **Two-stage:** regime model → allocators (one-stage kept only as control A1) | Project objective | **Directed** |
| DL-18 | — | Curriculum = fixed-order sampling schedule (not a new reward); same for all agents; control A2 | Keeps the comparison clean | Proposed |
| DL-19 | 252-day episodes | 126-day episodes | One 253-day COVID window would otherwise give almost one episode | Proposed |
| DL-20 | 5 seeds, tuned step counts | 3 seeds, fixed 150k steps, ordered time-box (TU-05) | Deadline | Proposed |
| DL-21 | No success definition | Pre-registered SC-1…SC-4 and phase windows | Prevents post-hoc goalposts | Proposed |
| DL-22 | 63-day rolling window | 21-day rolling window; CAGR blank under 252 days | War window is ≈ 44 days | Proposed |
| DL-23 | Test feature-shift report before freeze | Test-window feature/regime reports only after Step 23 | Looking earlier risks tuning to test | Proposed |
| DL-24 | T-16 on real data | T-16 on the formula with synthetic inputs | Real-data assertion can fail without a bug | Proposed |
| DL-25 | — | 2025-01-01 → 2026-02-26 excluded | Tariff crash contaminates "normal"; would leak into 2024 test | Proposed |
| DL-26 | — | Headline-model rule TU-04 (median Sortino on `val_2022`, guard first) | Answers "how do we pick the model" without touching test | Proposed |

### 21.2 Open Questions (answer, then move to the Decision Log)

| ID | Question | Default until answered |
|---|---|---|
| OQ-01 | Should cash earn a T-bill yield? | No (0 %). |
| OQ-02 | Add SPY as a market reference line on the dashboard? | No. |
| OQ-03 | Bootstrap confidence intervals for Sharpe differences? | Optional / stretch. |
| OQ-04 | Is next-open execution required as a robustness check? | Optional. |
| OQ-05 | Keep the one-stage control A1 (needed to test H4)? | Keep, droppable under TU-05. |
| OQ-06 | Extend `test_war` beyond 2026-04-30? | No — fixed (GR-10). |
| OQ-07 | Add 2025 tariff shock as an extra test scenario? | No. |
| OQ-08 | Add a regime-aware MVO benchmark? | No. |
| OQ-09 | If CU-02's difficulty table ranks windows differently, reorder stages? | Raise it here first; no silent reorder. |

### 21.3 Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-11-06 | First spec (single-stage, test 2020–2023). |
| 1.1 | 2026-11-06 | Two-stage design, curriculum, new split and scenarios, success criteria, quick tuning and time-box, 21-day rolling window, table-format fix. |

---

## 22. Change Process

1. Open a PR that edits `specs.md`, adds a Decision Log row, adds a Changelog row and bumps the version in the header.
2. Update `TODO.md` (affected steps and Done-when criteria) in the same PR.
3. At least one other teammate approves. No silent deviations — code that contradicts this file is a bug (GR-1).
4. After `config/FROZEN` exists, any change to §4–§15, `config.yaml`, scenario windows or success criteria invalidates the freeze and all test results; re-run Steps 22–25 and disclose it (GR-10).
