"""K-03 1/N: rebalance to 0.2 per stock every ``rebalance_every`` trading days."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.benchmarks.simulator import MarketData, simulate

EQUAL_WEIGHTS = np.array([0.2] * 5 + [0.0], dtype=np.float64)


def run_equal_weight(
    market: MarketData,
    first: int,
    last: int,
    *,
    rebalance_every: int = 21,
    **kwargs,
) -> pd.DataFrame:
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be positive")

    def target(step: int, index: int, weights: np.ndarray) -> np.ndarray | None:
        # The portfolio starts at 1/N, so the first rebalance is after one period.
        if step == 0 or step % rebalance_every:
            return None
        return EQUAL_WEIGHTS

    return simulate(market, first, last, target, **kwargs)
