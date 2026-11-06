"""Pure portfolio trade and mark-to-market operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.env.costs import CostModel

N_RISKY_ASSETS = 5
N_ASSETS = 6
WEIGHT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class TradeResult:
    """Portfolio value after a trade, with its cost and traded turnover."""

    holdings: NDArray[np.float64]
    cost: float
    turnover: float

    @property
    def weights(self) -> NDArray[np.float64]:
        return self.holdings / self.holdings.sum()


def _validate_holdings(holdings: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(holdings, dtype=float)
    if values.shape != (N_ASSETS,):
        raise ValueError("holdings must contain five risky assets and cash")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("holdings must be finite and non-negative")
    if values.sum() <= 0:
        raise ValueError("holdings must have positive total value")
    return values


def _validate_weights(target_weights: ArrayLike) -> NDArray[np.float64]:
    weights = np.asarray(target_weights, dtype=float)
    if weights.shape != (N_ASSETS,):
        raise ValueError("target_weights must contain five risky assets and cash")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("target_weights must be finite and non-negative")
    if abs(float(weights.sum()) - 1.0) >= WEIGHT_TOLERANCE:
        raise ValueError("target_weights must sum to 1 within 1e-9")
    return weights


def apply_trades(
    holdings: ArrayLike,
    target_weights: ArrayLike,
    cost_model: CostModel,
    *,
    volatility: ArrayLike | None = None,
) -> TradeResult:
    """Rebalance holdings and deduct fees while preserving target weights.

    Trading costs are charged on changes in the risky-asset holdings only.
    Fees are deducted proportionally from all holdings, leaving the resulting
    portfolio at the requested weights.
    """
    current = _validate_holdings(holdings)
    weights = _validate_weights(target_weights)
    if not isinstance(cost_model, CostModel):
        raise TypeError("cost_model must be a CostModel")

    portfolio_value = float(current.sum())
    target_holdings = weights * portfolio_value
    differences = target_holdings[:N_RISKY_ASSETS] - current[:N_RISKY_ASSETS]
    if np.max(np.abs(differences)) <= 1e-15 * portfolio_value:
        return TradeResult(current.copy(), 0.0, 0.0)
    traded_notional = np.abs(differences)
    cost = float(np.dot(traded_notional, cost_model.rates(volatility)))
    if cost >= portfolio_value:
        raise ValueError("trade costs must be less than the portfolio value")

    post_trade_holdings = weights * (portfolio_value - cost)
    turnover = float(traded_notional.sum() / portfolio_value)
    return TradeResult(post_trade_holdings, cost, turnover)


def mark_to_market(holdings: ArrayLike, returns: ArrayLike) -> NDArray[np.float64]:
    """Apply one period of simple returns; the cash bucket earns zero."""
    current = _validate_holdings(holdings)
    asset_returns = np.asarray(returns, dtype=float)
    if asset_returns.shape != (N_RISKY_ASSETS,):
        raise ValueError("returns must contain one simple return for each risky asset")
    if not np.isfinite(asset_returns).all() or (asset_returns < -1).any():
        raise ValueError("returns must be finite and no less than -1")
    updated = current.copy()
    updated[:N_RISKY_ASSETS] *= 1.0 + asset_returns
    return updated
