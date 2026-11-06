import numpy as np

from src.env.costs import CostModel
from src.env.portfolio import apply_trades, mark_to_market


def test_t08_always_hold_matches_buy_and_hold_equity_curve():
    daily_returns = np.random.default_rng(8).normal(0.0005, 0.02, size=(100, 5))
    initial = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
    hold_holdings = buy_and_hold_holdings = initial.copy()
    cost_model = CostModel()

    for returns in daily_returns:
        hold_weights = hold_holdings / hold_holdings.sum()
        trade = apply_trades(hold_holdings, hold_weights, cost_model)
        assert trade.cost == 0.0
        hold_holdings = mark_to_market(trade.holdings, returns)
        buy_and_hold_holdings = mark_to_market(buy_and_hold_holdings, returns)
        assert abs(float(hold_holdings.sum()) - float(buy_and_hold_holdings.sum())) < 1e-9
