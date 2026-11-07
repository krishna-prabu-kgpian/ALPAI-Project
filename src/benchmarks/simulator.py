"""Shared benchmark simulator (K-01) built on the environment's trade primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.agents.curriculum import load_config
from src.env.costs import CostModel
from src.env.portfolio import apply_trades, mark_to_market

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TICKERS = ("NVDA", "AAPL", "MSFT", "GOOGL", "AMZN")
BUCKETS = (*TICKERS, "CASH")
RECORD_COLUMNS = (
    "date",
    "portfolio_value",
    "daily_return",
    "cost_paid",
    "turnover",
    *(f"w_{name}" for name in BUCKETS),
    "p_calm",
    "p_choppy",
    "p_crisis",
)
FIT_LABELS = {"final": "2012-2023", "tuning": "2012-2021"}

# target_fn(step, index, pre_trade_weights) -> target weights, or None to hold
TargetFunction = Callable[[int, int, NDArray[np.float64]], NDArray[np.float64] | None]


@dataclass(frozen=True)
class MarketData:
    """Date-aligned inputs shared by benchmarks and the RL backtest."""

    dates: pd.DatetimeIndex
    asset_returns: NDArray[np.float64]
    regime_probabilities: NDArray[np.float64]
    history_returns: pd.DataFrame

    def window_bounds(self, window: str, config: dict[str, Any] | None = None) -> tuple[int, int]:
        """Return (first, last) row indices of a single-range config window."""
        windows = (load_config() if config is None else config)["windows"]
        start, end = (pd.Timestamp(value) for value in windows[window])
        selected = np.flatnonzero((self.dates >= start) & (self.dates <= end))
        if selected.size < 2:
            raise ValueError(f"window {window!r} must contain at least two dates")
        return int(selected[0]), int(selected[-1])


def load_market_data(fit: str = "final") -> MarketData:
    """Load returns and regime probabilities aligned with the feature dates."""
    label = FIT_LABELS[fit]
    processed = PROJECT_ROOT / "data" / "processed"
    features = pd.read_parquet(processed / f"features_z_{label}.parquet")
    regimes = pd.read_parquet(processed / f"regime_probs_{label}.parquet")
    if not features.index.equals(regimes.index):
        raise ValueError("feature and regime dates must match")
    prices = pd.read_parquet(processed / "prices_clean.parquet")
    close = prices.xs("Close", axis=1, level=1).reindex(columns=list(TICKERS))
    history = close.pct_change(fill_method=None)
    returns = history.reindex(features.index)
    if returns.isna().any().any():
        raise ValueError("clean prices do not cover every feature date")
    return MarketData(
        dates=pd.DatetimeIndex(features.index),
        asset_returns=returns.to_numpy(dtype=np.float64),
        regime_probabilities=regimes.to_numpy(dtype=np.float64)[:, :3],
        history_returns=history,
    )


def default_cost_model(config: dict[str, Any] | None = None) -> CostModel:
    env_config = (load_config() if config is None else config)["env"]
    return CostModel(cost_rate=float(env_config["cost"]), slippage_mode="none")


def simulate(
    market: MarketData,
    first: int,
    last: int,
    target_fn: TargetFunction,
    *,
    cost_model: CostModel | None = None,
    initial_value: float = 1.0,
) -> pd.DataFrame:
    """Trade at the close of each day in [first, last), hold to the next close.

    Mirrors PortfolioEnv.step (E-02): one output row per next-day close, so a
    window of n dates yields n - 1 rows, exactly like a greedy env pass.
    """
    if not 0 <= first < last < len(market.dates):
        raise ValueError("window bounds must satisfy 0 <= first < last < len(dates)")
    cost_model = default_cost_model() if cost_model is None else cost_model
    holdings = initial_value * np.array([0.2] * 5 + [0.0], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for step, index in enumerate(range(first, last)):
        value_before = float(holdings.sum())
        weights = holdings / value_before
        target = target_fn(step, index, weights.copy())
        if target is None:
            cost, turnover = 0.0, 0.0
        else:
            trade = apply_trades(holdings, target, cost_model)
            holdings, cost, turnover = trade.holdings, trade.cost, trade.turnover
        holdings = mark_to_market(holdings, market.asset_returns[index + 1])
        value_after = float(holdings.sum())
        rows.append(
            record_row(
                market.dates[index + 1],
                value_after,
                value_after / value_before - 1.0,
                cost,
                turnover,
                holdings / value_after,
                market.regime_probabilities[index + 1],
            )
        )
    return pd.DataFrame(rows, columns=list(RECORD_COLUMNS))


def record_row(
    date: pd.Timestamp,
    value: float,
    daily_return: float,
    cost: float,
    turnover: float,
    weights: NDArray[np.float64],
    regime_probabilities: NDArray[np.float64],
) -> dict[str, Any]:
    """One C-01 runner row (the first 14 columns)."""
    row: dict[str, Any] = {
        "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
        "portfolio_value": float(value),
        "daily_return": float(daily_return),
        "cost_paid": float(cost),
        "turnover": float(turnover),
    }
    row.update({f"w_{name}": float(w) for name, w in zip(BUCKETS, weights)})
    row.update(
        {
            name: float(p)
            for name, p in zip(("p_calm", "p_choppy", "p_crisis"), regime_probabilities)
        }
    )
    return row
