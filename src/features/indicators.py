"""Causal, trailing-window indicators for adjusted asset prices."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_ORDER = (
    "ret_1d",
    "sma10_gap",
    "sma50_gap",
    "rsi14",
    "vol20",
    "vol_chg",
)


def compute_indicators(
    prices: pd.DataFrame,
    *,
    sma_short: int = 10,
    sma_long: int = 50,
    rsi_period: int = 14,
    vol_window: int = 20,
    volume_window: int = 20,
) -> pd.DataFrame:
    """Return six causal features per ticker from adjusted Close and Volume.

    The input columns must be a ``(ticker, field)`` MultiIndex. Rolling
    windows require a full window; leading values remain NaN for the
    feature-building pipeline to drop consistently.
    """
    _validate_prices(prices)
    window_values = (sma_short, sma_long, rsi_period, vol_window, volume_window)
    if any(not isinstance(window, int) or window < 1 for window in window_values):
        raise ValueError("Indicator window sizes must be positive integers")

    tickers = list(dict.fromkeys(prices.columns.get_level_values("ticker")))
    features: dict[tuple[str, str], pd.Series] = {}
    annualization = np.sqrt(252.0)

    for ticker in tickers:
        close = prices[(ticker, "Close")].astype(float)
        volume = prices[(ticker, "Volume")].astype(float)
        log_return = np.log(close / close.shift(1))

        features[(ticker, "ret_1d")] = log_return
        features[(ticker, "sma10_gap")] = (
            close / close.rolling(sma_short, min_periods=sma_short).mean() - 1.0
        )
        features[(ticker, "sma50_gap")] = (
            close / close.rolling(sma_long, min_periods=sma_long).mean() - 1.0
        )
        features[(ticker, "rsi14")] = _wilder_rsi(close, rsi_period)
        features[(ticker, "vol20")] = (
            log_return.rolling(vol_window, min_periods=vol_window).std(ddof=1)
            * annualization
        )
        rolling_volume = volume.rolling(
            volume_window, min_periods=volume_window
        ).mean()
        features[(ticker, "vol_chg")] = np.log(
            (volume + 1.0) / (rolling_volume + 1.0)
        )

    columns = pd.MultiIndex.from_tuples(
        features.keys(), names=["asset", "feature"]
    )
    result = pd.DataFrame(features, index=prices.index)
    result.columns = columns
    ordered_columns = pd.MultiIndex.from_product(
        [tickers, FEATURE_ORDER], names=["asset", "feature"]
    )
    return result.reindex(columns=ordered_columns)


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = gains.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    average_loss = losses.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()

    relative_strength = average_gain / average_loss
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    no_losses = average_loss.eq(0)
    no_gains = average_gain.eq(0)
    rsi = rsi.mask(no_losses, 100.0)
    rsi = rsi.mask(no_losses & no_gains, 50.0)
    return rsi


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
    required = pd.MultiIndex.from_product(
        [
            list(dict.fromkeys(prices.columns.get_level_values("ticker"))),
            ["Close", "Volume"],
        ],
        names=["ticker", "field"],
    )
    missing = required.difference(prices.columns)
    if len(missing):
        raise ValueError(f"Prices are missing required columns: {list(missing)}")
    close = prices.loc[:, pd.IndexSlice[:, "Close"]].to_numpy(dtype=float)
    volume = prices.loc[:, pd.IndexSlice[:, "Volume"]].to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Adjusted close values must be finite and positive")
    if not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError("Volume values must be finite and non-negative")
