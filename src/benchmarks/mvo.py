"""K-04 Mean-Variance: trailing max-Sharpe weights via SciPy SLSQP."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.benchmarks.simulator import MarketData, simulate

LOGGER = logging.getLogger(__name__)
RIDGE = 1e-6


def max_sharpe_weights(
    trailing_returns: NDArray[np.float64],
    initial_guess: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], bool]:
    """Long-only, fully invested max-Sharpe weights (risk-free 0) and the success flag."""
    returns = np.asarray(trailing_returns, dtype=np.float64)
    if returns.ndim != 2 or returns.shape[1] != 5 or not np.isfinite(returns).all():
        raise ValueError("trailing_returns must be a finite (days, 5) array")
    mean = returns.mean(axis=0)
    covariance = np.cov(returns, rowvar=False, ddof=1) + RIDGE * np.eye(5)
    guess = np.full(5, 0.2) if initial_guess is None else np.asarray(initial_guess, dtype=float)

    def negative_sharpe(weights: NDArray[np.float64]) -> float:
        return -float(weights @ mean) / float(np.sqrt(weights @ covariance @ weights))

    result = minimize(
        negative_sharpe,
        guess,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 5,
        constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}],
    )
    weights = np.clip(result.x, 0.0, 1.0)
    weights /= weights.sum()
    return weights, bool(result.success)


def run_mvo(
    market: MarketData,
    first: int,
    last: int,
    *,
    window: int = 252,
    rebalance_every: int = 21,
    **kwargs,
) -> pd.DataFrame:
    history = market.history_returns

    def target(step: int, index: int, weights: np.ndarray) -> np.ndarray | None:
        if step % rebalance_every:
            return None
        end = history.index.get_loc(market.dates[index])
        trailing = history.iloc[end - window + 1 : end + 1].to_numpy()
        if len(trailing) < window or np.isnan(trailing).any():
            raise ValueError(f"insufficient {window}-day history before {market.dates[index]}")
        optimal, success = max_sharpe_weights(trailing)
        if not success:
            LOGGER.warning("SLSQP failed on %s; keeping previous weights", market.dates[index])
            return None
        return np.append(optimal, 0.0)

    return simulate(market, first, last, target, **kwargs)
