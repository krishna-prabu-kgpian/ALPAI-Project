"""Single implementation of the §14 performance and risk metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

PERIODS_PER_YEAR = 252
ROLLING_WINDOW = 21
CVAR_LEVEL = 0.05
METRIC_NAMES = (
    "total_return",
    "cagr",
    "ann_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "cvar_5",
    "avg_turnover",
    "total_cost",
)


def _as_returns(daily_returns: ArrayLike) -> NDArray[np.float64]:
    returns = np.asarray(daily_returns, dtype=np.float64).ravel()
    if returns.size == 0:
        raise ValueError("daily_returns must contain at least one value")
    if not np.isfinite(returns).all() or (returns < -1).any():
        raise ValueError("daily_returns must be finite and no less than -1")
    return returns


def equity_curve(daily_returns: ArrayLike, initial_value: float = 1.0) -> NDArray[np.float64]:
    """Return the value path ``[V0, V1, ..., Vn]`` implied by net daily returns."""
    returns = _as_returns(daily_returns)
    return initial_value * np.concatenate(([1.0], np.cumprod(1.0 + returns)))


def total_return(daily_returns: ArrayLike) -> float:
    """X-01: compounded net return over the window."""
    return float(np.prod(1.0 + _as_returns(daily_returns)) - 1.0)


def cagr(daily_returns: ArrayLike, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """X-01: annualized growth; NaN for windows shorter than one year."""
    returns = _as_returns(daily_returns)
    if returns.size < periods_per_year:
        return math.nan
    growth = 1.0 + total_return(returns)
    return float(growth ** (periods_per_year / returns.size) - 1.0)


def annualized_volatility(daily_returns: ArrayLike, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """X-02: sample standard deviation of daily returns times sqrt(252)."""
    returns = _as_returns(daily_returns)
    if returns.size < 2:
        return math.nan
    return float(np.std(returns, ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(daily_returns: ArrayLike, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """X-02: mean / std times sqrt(252), risk-free rate 0."""
    returns = _as_returns(daily_returns)
    if returns.size < 2:
        return math.nan
    deviation = float(np.std(returns, ddof=1))
    if deviation == 0.0:
        return math.nan
    return float(np.mean(returns) / deviation * math.sqrt(periods_per_year))


def downside_deviation(daily_returns: ArrayLike) -> float:
    """Daily semi-deviation below a zero target: sqrt(mean(min(r, 0)^2))."""
    returns = _as_returns(daily_returns)
    return float(math.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))


def sortino_ratio(daily_returns: ArrayLike, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """X-03: annualized mean over annualized downside deviation, target 0."""
    returns = _as_returns(daily_returns)
    downside = downside_deviation(returns)
    if downside == 0.0:
        return math.nan
    return float(
        np.mean(returns) * periods_per_year / (downside * math.sqrt(periods_per_year))
    )


def max_drawdown(daily_returns: ArrayLike) -> float:
    """X-04: largest peak-to-trough decline of the equity path, as a positive fraction."""
    values = equity_curve(daily_returns)
    return float(np.max(1.0 - values / np.maximum.accumulate(values)))


def calmar_ratio(daily_returns: ArrayLike, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """X-04: CAGR / max drawdown; NaN whenever CAGR is blank or drawdown is zero."""
    growth = cagr(daily_returns, periods_per_year)
    drawdown = max_drawdown(daily_returns)
    if math.isnan(growth) or drawdown == 0.0:
        return math.nan
    return float(growth / drawdown)


def cvar(daily_returns: ArrayLike, level: float = CVAR_LEVEL) -> float:
    """X-05: mean of the daily returns at or below the historical ``level`` quantile."""
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    returns = _as_returns(daily_returns)
    threshold = np.quantile(returns, level)
    return float(np.mean(returns[returns <= threshold]))


def compute_metrics(
    daily_returns: ArrayLike,
    turnover: ArrayLike | None = None,
    cost_paid: ArrayLike | None = None,
    *,
    initial_value: float = 1.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[str, float]:
    """Return every C-02 metric for one strategy over one window.

    ``turnover`` is the daily sum of absolute risky-asset weight changes and
    ``cost_paid`` the daily cost in currency; both are optional (NaN if absent).
    """
    returns = _as_returns(daily_returns)
    turnover_values = None if turnover is None else np.asarray(turnover, dtype=np.float64)
    cost_values = None if cost_paid is None else np.asarray(cost_paid, dtype=np.float64)
    for name, values in (("turnover", turnover_values), ("cost_paid", cost_values)):
        if values is not None and (values.shape != returns.shape or not np.isfinite(values).all()):
            raise ValueError(f"{name} must be finite and aligned with daily_returns")
    return {
        "total_return": total_return(returns),
        "cagr": cagr(returns, periods_per_year),
        "ann_vol": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
        "cvar_5": cvar(returns),
        "avg_turnover": math.nan if turnover_values is None else float(np.mean(turnover_values)),
        "total_cost": (
            math.nan if cost_values is None else float(np.sum(cost_values) / initial_value)
        ),
    }


def rolling_sharpe(
    daily_returns: ArrayLike,
    window: int = ROLLING_WINDOW,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> NDArray[np.float64]:
    """X-07: trailing Sharpe; the first ``window - 1`` rows are NaN."""
    returns = pd.Series(_as_returns(daily_returns))
    rolling = returns.rolling(window, min_periods=window)
    deviation = rolling.std(ddof=1).replace(0.0, np.nan)
    return (rolling.mean() / deviation * math.sqrt(periods_per_year)).to_numpy()


def rolling_sortino(
    daily_returns: ArrayLike,
    window: int = ROLLING_WINDOW,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> NDArray[np.float64]:
    """X-07: trailing Sortino; the first ``window - 1`` rows are NaN."""
    returns = pd.Series(_as_returns(daily_returns))
    mean = returns.rolling(window, min_periods=window).mean()
    downside = (
        (np.minimum(returns, 0.0) ** 2).rolling(window, min_periods=window).mean() ** 0.5
    ).replace(0.0, np.nan)
    return (mean * periods_per_year / (downside * math.sqrt(periods_per_year))).to_numpy()


def drawdown_series(daily_returns: ArrayLike) -> NDArray[np.float64]:
    """Drawdown after each day's return, aligned with ``daily_returns``."""
    values = equity_curve(daily_returns)
    return (1.0 - values / np.maximum.accumulate(values))[1:]
