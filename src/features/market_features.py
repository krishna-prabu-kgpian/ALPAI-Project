"""Causal market-level basket features."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

MARKET_FEATURE_ORDER = ("m_vol20", "m_corr20", "m_dd")


def compute_market_features(
    prices: pd.DataFrame,
    *,
    vol_window: int = 20,
    corr_window: int = 20,
    drawdown_window: int = 252,
) -> pd.DataFrame:
    """Compute basket volatility, mean asset correlation, and drawdown."""
    _validate_prices(prices)
    windows = (vol_window, corr_window, drawdown_window)
    if any(not isinstance(window, int) or window < 1 for window in windows):
        raise ValueError("Market feature window sizes must be positive integers")

    tickers = list(dict.fromkeys(prices.columns.get_level_values("ticker")))
    if len(tickers) < 2:
        raise ValueError("Market correlation requires at least two assets")

    close = pd.DataFrame(
        {ticker: prices[(ticker, "Close")].astype(float) for ticker in tickers},
        index=prices.index,
    )
    simple_returns = close.pct_change(fill_method=None)
    log_returns = np.log(close / close.shift(1))
    basket_returns = simple_returns.mean(axis=1)
    market_volatility = (
        basket_returns.rolling(vol_window, min_periods=vol_window).std(ddof=1)
        * np.sqrt(252.0)
    )

    pair_correlations = []
    for left, right in itertools.combinations(tickers, 2):
        correlation = log_returns[left].rolling(
            corr_window, min_periods=corr_window
        ).corr(log_returns[right])
        window_ready = (
            log_returns[left].rolling(
                corr_window, min_periods=corr_window
            ).count().eq(corr_window)
            & log_returns[right].rolling(
                corr_window, min_periods=corr_window
            ).count().eq(corr_window)
        )
        pair_correlations.append(correlation.mask(window_ready & correlation.isna(), 0.0))
    market_correlation = pd.concat(pair_correlations, axis=1).mean(axis=1)

    # The first day's return is undefined; treating it as zero establishes I_0=1.
    basket_index = (1.0 + basket_returns.fillna(0.0)).cumprod()
    rolling_peak = basket_index.rolling(
        drawdown_window, min_periods=1
    ).max()
    market_drawdown = 1.0 - basket_index / rolling_peak

    result = pd.DataFrame(
        {
            "m_vol20": market_volatility,
            "m_corr20": market_correlation,
            "m_dd": market_drawdown,
        },
        index=prices.index,
    )
    result.columns.name = "feature"
    return result.reindex(columns=MARKET_FEATURE_ORDER)


def _validate_prices(prices: pd.DataFrame) -> None:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Prices must be a non-empty pandas DataFrame")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Price index must be a DatetimeIndex")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and strictly increasing")
    if (
        not isinstance(prices.columns, pd.MultiIndex)
        or prices.columns.nlevels != 2
        or "ticker" not in prices.columns.names
        or "field" not in prices.columns.names
    ):
        raise ValueError("Price columns must be a (ticker, field) MultiIndex")
    tickers = list(dict.fromkeys(prices.columns.get_level_values("ticker")))
    required = pd.MultiIndex.from_product(
        [tickers, ["Close"]], names=["ticker", "field"]
    )
    missing = required.difference(prices.columns)
    if len(missing):
        raise ValueError(f"Prices are missing adjusted close columns: {list(missing)}")
    close = prices.loc[:, pd.IndexSlice[:, "Close"]].to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Adjusted close values must be finite and positive")
