"""K-02 Buy & Hold: start equal-weight and never trade."""

from __future__ import annotations

import pandas as pd

from src.benchmarks.simulator import MarketData, simulate


def run_buy_hold(market: MarketData, first: int, last: int, **kwargs) -> pd.DataFrame:
    return simulate(market, first, last, lambda step, index, weights: None, **kwargs)
