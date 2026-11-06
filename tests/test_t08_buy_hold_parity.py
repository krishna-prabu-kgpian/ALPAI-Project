import numpy as np

from src.env.portfolio import mark_to_market
from src.env.portfolio_env import PortfolioEnv


def test_t08_always_hold_matches_buy_and_hold_equity_curve():
    daily_returns = np.random.default_rng(8).normal(0.0005, 0.02, size=(100, 5))
    initial = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
    features = np.zeros((101, 30))
    regimes = np.tile([0.2, 0.3, 0.5], (101, 1))
    asset_returns = np.vstack([np.zeros((1, 5)), daily_returns])
    env = PortfolioEnv(
        features,
        regimes,
        asset_returns=asset_returns,
        length=100,
        cost=0.001,
        lambda1=0.0,
        lambda2=0.0,
    )
    env.reset(seed=8)
    buy_and_hold_holdings = initial.copy()

    for returns in daily_returns:
        _, _, _, _, info = env.step(0)
        assert info["cost"] == 0.0
        buy_and_hold_holdings = mark_to_market(buy_and_hold_holdings, returns)
        assert abs(float(info["V"]) - float(buy_and_hold_holdings.sum())) < 1e-9
