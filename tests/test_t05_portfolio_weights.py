import numpy as np

from src.env.costs import CostModel
from src.env.portfolio import apply_trades


def test_t05_random_trades_preserve_long_only_weights():
    rng = np.random.default_rng(2026)
    holdings = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
    cost_model = CostModel()

    for _ in range(10_000):
        weights = rng.dirichlet(np.ones(6))
        result = apply_trades(holdings, weights, cost_model)
        assert np.all(result.holdings >= 0)
        assert abs(float(result.weights.sum()) - 1.0) < 1e-9
        holdings = result.holdings
