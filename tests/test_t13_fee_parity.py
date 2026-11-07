import numpy as np
import pandas as pd
import pytest

from src.benchmarks.buy_hold import run_buy_hold
from src.benchmarks.equal_weight import run_equal_weight
from src.benchmarks.mvo import max_sharpe_weights
from src.benchmarks.simulator import MarketData, simulate
from src.env.portfolio_env import PortfolioEnv, decode_action

N_DAYS = 60


def _synthetic_market(seed: int = 13) -> MarketData:
    rng = np.random.default_rng(seed)
    returns = np.vstack([np.zeros((1, 5)), rng.normal(0.0005, 0.02, size=(N_DAYS, 5))])
    dates = pd.bdate_range("2030-01-01", periods=N_DAYS + 1)
    regimes = np.tile([0.5, 0.3, 0.2], (N_DAYS + 1, 1))
    return MarketData(dates, returns, regimes, pd.DataFrame(returns, index=dates))


def _env(market: MarketData) -> PortfolioEnv:
    return PortfolioEnv(
        np.zeros((N_DAYS + 1, 30)),
        market.regime_probabilities,
        asset_returns=market.asset_returns,
        length=N_DAYS,
        lambda1=0.0,
        lambda2=0.0,
    )


def _transfer(weights: np.ndarray, action_id: int) -> np.ndarray | None:
    action = decode_action(action_id)
    if action["from"] is None or weights[action["from"]] < 1e-9:
        return None
    target = weights.copy()
    amount = min(0.10, float(weights[action["from"]]))
    target[action["from"]] -= amount
    target[action["to"]] += amount
    return target


def test_t13_fixed_target_rebalance_matches_env_equity_curve():
    market = _synthetic_market()
    # Two trades out of NVDA (one into AAPL, one into cash), then cash -> MSFT, then hold.
    schedule = {0: 1, 1: 5, 7: 28}
    env = _env(market)
    env.reset(seed=13)
    env_values, env_costs = [], []
    for step in range(N_DAYS):
        _, _, _, _, info = env.step(schedule.get(step, 0))
        env_values.append(info["V"])
        env_costs.append(info["cost"])

    frame = simulate(
        market,
        0,
        N_DAYS,
        lambda step, index, weights: (
            _transfer(weights, schedule[step]) if step in schedule else None
        ),
    )
    np.testing.assert_allclose(frame["portfolio_value"], env_values, rtol=0, atol=1e-12)
    np.testing.assert_allclose(frame["cost_paid"], env_costs, rtol=0, atol=1e-15)
    assert frame["cost_paid"].iloc[0] == pytest.approx(0.001 * 0.2 * 1.0)


def test_buy_hold_matches_always_hold_env_and_pays_nothing():
    market = _synthetic_market(8)
    env = _env(market)
    env.reset(seed=8)
    values = [env.step(0)[4]["V"] for _ in range(N_DAYS)]
    frame = run_buy_hold(market, 0, N_DAYS)
    np.testing.assert_allclose(frame["portfolio_value"], values, rtol=0, atol=1e-12)
    assert frame["cost_paid"].sum() == 0.0
    assert list(frame.columns)[:5] == [
        "date",
        "portfolio_value",
        "daily_return",
        "cost_paid",
        "turnover",
    ]


def test_equal_weight_rebalances_every_21_days_back_to_one_fifth():
    market = _synthetic_market(3)
    frame = run_equal_weight(market, 0, N_DAYS, rebalance_every=21)
    traded = np.flatnonzero(frame["cost_paid"].to_numpy() > 0)
    np.testing.assert_array_equal(traded, [21, 42])
    # After a rebalance, the drift-free part of the day starts from 0.2 each.
    weights_after = frame.loc[21, [f"w_{t}" for t in ("NVDA", "AAPL", "MSFT", "GOOGL", "AMZN")]].to_numpy(float)
    returns = market.asset_returns[22]
    expected = 0.2 * (1 + returns) / (0.2 * (1 + returns)).sum()
    np.testing.assert_allclose(weights_after, expected, atol=1e-12)


def test_mvo_weights_are_long_only_and_favor_the_better_asset():
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0, 0.01, size=(252, 5))
    returns[:, 2] += 0.003
    weights, success = max_sharpe_weights(returns)
    assert success
    assert weights.min() >= 0.0 and weights.sum() == pytest.approx(1.0)
    assert int(np.argmax(weights)) == 2
