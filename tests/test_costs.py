import numpy as np
import pytest

from src.env.costs import CostModel
from src.env.portfolio import apply_trades, mark_to_market


def test_t06_holding_with_drifting_weights_has_zero_cost():
    holdings = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
    cost_model = CostModel()
    total_cost = 0.0

    for _ in range(100):
        weights = holdings / holdings.sum()
        trade = apply_trades(holdings, weights, cost_model)
        total_cost += trade.cost
        holdings = mark_to_market(
            trade.holdings, np.array([0.01, -0.02, 0.005, 0.0, 0.03])
        )

    assert total_cost == 0.0


def test_t07_hand_computed_risky_and_cash_trade_costs():
    cost_model = CostModel()
    initial = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
    risky_trade = apply_trades(
        initial, np.array([0.1, 0.3, 0.2, 0.2, 0.2, 0.0]), cost_model
    )
    assert risky_trade.cost == pytest.approx(0.0002)
    assert risky_trade.holdings.sum() == pytest.approx(0.9998)
    np.testing.assert_allclose(
        risky_trade.weights, [0.1, 0.3, 0.2, 0.2, 0.2, 0.0], atol=1e-12
    )

    cash_trade = apply_trades(
        np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1]),
        np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0]),
        cost_model,
    )
    assert cash_trade.cost == pytest.approx(0.0001)
    assert cash_trade.holdings.sum() == pytest.approx(0.9999)


def test_t16_volatility_scaled_slippage_formula_and_bounds():
    model = CostModel(
        slippage_mode="vol_scaled",
        slippage_s0_bps=2.0,
        median_fit_volatility=[0.01, 0.02, 0.03, 0.04, 0.05],
    )
    current_volatility = np.array([0.005, 0.02, 0.09, 0.4, 0.5])
    np.testing.assert_allclose(
        model.slippage_bps(current_volatility), [2.0, 2.0, 6.0, 10.0, 10.0]
    )
    rates = model.rates(current_volatility)
    assert np.all(rates >= model.cost_rate + 2.0 / 10_000)
    assert np.all(rates <= model.cost_rate + 10.0 / 10_000)


@pytest.mark.parametrize(
    ("holdings", "weights"),
    [
        ([0.2] * 5 + [0.0], [0.2] * 5 + [-1e-3]),
        ([0.2] * 5 + [0.0], [0.2] * 5 + [0.1]),
        ([0.2] * 5, [0.2] * 5 + [0.0]),
    ],
)
def test_invalid_trade_inputs_are_rejected(holdings, weights):
    with pytest.raises(ValueError):
        apply_trades(holdings, weights, CostModel())
