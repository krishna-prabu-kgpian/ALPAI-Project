import math

import numpy as np
import pytest

from src.evaluation.metrics import (
    cagr,
    calmar_ratio,
    compute_metrics,
    cvar,
    drawdown_series,
    max_drawdown,
    rolling_sharpe,
    rolling_sortino,
)

SHORT_SERIES = [0.10, -0.05, 0.02, -0.01]


def test_t14_metrics_match_hand_computed_short_series():
    metrics = compute_metrics(
        SHORT_SERIES,
        turnover=[0.2, 0.0, 0.1, 0.1],
        cost_paid=[0.0002, 0.0, 0.0001, 0.0001],
    )

    # 1.10 * 0.95 * 1.02 * 0.99 = 1.055241
    assert metrics["total_return"] == pytest.approx(0.055241, abs=1e-12)
    # mean 0.015; sample std of the four values
    sample_std = math.sqrt(
        sum((r - 0.015) ** 2 for r in SHORT_SERIES) / 3
    )
    assert metrics["ann_vol"] == pytest.approx(sample_std * math.sqrt(252))
    assert metrics["sharpe"] == pytest.approx(0.015 / sample_std * math.sqrt(252))
    # downside deviation sqrt((0.05^2 + 0.01^2) / 4) = sqrt(0.00065)
    assert metrics["sortino"] == pytest.approx(
        0.015 * 252 / (math.sqrt(0.00065) * math.sqrt(252))
    )
    # equity 1, 1.1, 1.045, ... -> peak 1.1, trough 1.045
    assert metrics["max_drawdown"] == pytest.approx(0.05)
    # 5th percentile of four values is -0.044, only -0.05 lies at or below it
    assert metrics["cvar_5"] == pytest.approx(-0.05)
    assert metrics["avg_turnover"] == pytest.approx(0.1)
    assert metrics["total_cost"] == pytest.approx(0.0004)
    # under 252 days: CAGR and Calmar are blank
    assert math.isnan(metrics["cagr"])
    assert math.isnan(metrics["calmar"])


def test_cagr_and_calmar_on_one_full_year():
    returns = np.tile([0.01, -0.005], 126)
    growth = (1.01 * 0.995) ** 126
    assert cagr(returns) == pytest.approx(growth - 1.0)
    assert max_drawdown(returns) == pytest.approx(0.005)
    assert calmar_ratio(returns) == pytest.approx((growth - 1.0) / 0.005)
    assert math.isnan(cagr(returns[:251]))


def test_cvar_averages_the_lower_tail():
    returns = np.arange(-50, 50) / 1000.0
    threshold = np.quantile(returns, 0.05)
    assert cvar(returns) == pytest.approx(returns[returns <= threshold].mean())
    assert cvar(returns) < np.quantile(returns, 0.05) + 1e-12


def test_rolling_metrics_use_21_day_window_with_blank_warmup():
    returns = np.random.default_rng(14).normal(0.001, 0.01, 40)
    sharpe = rolling_sharpe(returns)
    sortino = rolling_sortino(returns)
    assert np.isnan(sharpe[:20]).all() and np.isnan(sortino[:20]).all()
    window = returns[19:40]
    assert sharpe[39] == pytest.approx(
        window.mean() / window.std(ddof=1) * math.sqrt(252)
    )
    downside = math.sqrt(np.mean(np.minimum(window, 0.0) ** 2))
    assert sortino[39] == pytest.approx(window.mean() * 252 / (downside * math.sqrt(252)))


def test_drawdown_series_is_aligned_with_returns():
    np.testing.assert_allclose(
        drawdown_series(SHORT_SERIES), [0.0, 0.05, 0.031, 0.0406900], atol=1e-7
    )
