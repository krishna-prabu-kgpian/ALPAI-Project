import json

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features
from src.features.indicators import FEATURE_ORDER, compute_indicators
from src.features.market_features import compute_market_features


TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]
FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def _prices(rows=400):
    dates = pd.bdate_range("2012-01-03", periods=rows)
    columns = pd.MultiIndex.from_product(
        [TICKERS, FIELDS], names=["ticker", "field"]
    )
    index = np.arange(rows, dtype=float)
    data = {}
    for asset_index, ticker in enumerate(TICKERS):
        close = 30 + asset_index * 10 + index * (0.03 + asset_index * 0.002)
        close *= 1 + 0.015 * np.sin(index / (3.0 + asset_index))
        volume = 100_000 + asset_index * 1_000 + (index.astype(int) % 17) * 100
        data[(ticker, "Open")] = close * 0.999
        data[(ticker, "High")] = close * 1.01
        data[(ticker, "Low")] = close * 0.99
        data[(ticker, "Close")] = close
        data[(ticker, "Volume")] = volume
    frame = pd.DataFrame(data, index=dates).reindex(columns=columns)
    return frame


def test_indicators_match_spec_formulas():
    prices = _prices(80)
    features = compute_indicators(prices)
    close = prices[("NVDA", "Close")]
    volume = prices[("NVDA", "Volume")]
    returns = np.log(close / close.shift(1))

    pd.testing.assert_series_equal(
        features[("NVDA", "ret_1d")],
        returns,
        check_names=False,
    )
    expected_volatility = returns.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    pd.testing.assert_series_equal(
        features[("NVDA", "vol20")],
        expected_volatility,
        check_names=False,
    )
    expected_volume_change = np.log(
        (volume + 1) / (volume.rolling(20, min_periods=20).mean() + 1)
    )
    pd.testing.assert_series_equal(
        features[("NVDA", "vol_chg")],
        expected_volume_change,
        check_names=False,
    )
    assert tuple(dict.fromkeys(features.columns.get_level_values("feature"))) == FEATURE_ORDER


def test_market_features_match_basket_definitions():
    prices = _prices(80)
    market = compute_market_features(prices)
    close = prices.loc[:, pd.IndexSlice[:, "Close"]]
    simple_returns = close.pct_change(fill_method=None)
    basket_returns = simple_returns.mean(axis=1)

    expected_volatility = basket_returns.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252)
    pd.testing.assert_series_equal(
        market["m_vol20"],
        expected_volatility,
        check_names=False,
    )
    expected_index = (1 + basket_returns.fillna(0)).cumprod()
    expected_drawdown = 1 - expected_index / expected_index.rolling(
        252, min_periods=1
    ).max()
    pd.testing.assert_series_equal(
        market["m_dd"],
        expected_drawdown,
        check_names=False,
    )
    assert market["m_corr20"].iloc[:20].isna().all()
    assert np.isfinite(market["m_corr20"].iloc[20:]).all()


def test_market_correlation_uses_log_returns_and_zero_variance_counts_as_zero():
    prices = _prices(60)
    prices[("NVDA", "Close")] = 100.0
    prices[("AAPL", "Close")] = np.linspace(20.0, 40.0, len(prices))

    market = compute_market_features(prices)
    close = pd.DataFrame(
        {ticker: prices[(ticker, "Close")] for ticker in TICKERS},
        index=prices.index,
    )
    log_returns = np.log(close / close.shift(1))
    correlations = []
    tickers = list(dict.fromkeys(prices.columns.get_level_values("ticker")))
    for left_index, left in enumerate(tickers):
        for right in tickers[left_index + 1 :]:
            left_returns = log_returns[left].iloc[1:21]
            right_returns = log_returns[right].iloc[1:21]
            if left_returns.std(ddof=1) == 0 or right_returns.std(ddof=1) == 0:
                correlations.append(0.0)
            else:
                correlations.append(left_returns.corr(right_returns))

    assert market["m_corr20"].iloc[20] == pytest.approx(np.mean(correlations))
    assert any(value == 0.0 for value in correlations)


def test_wilder_rsi_handles_zero_loss_and_flat_series():
    prices = _prices(40)
    prices[("NVDA", "Close")] = np.arange(40, dtype=float) + 10
    prices[("AAPL", "Close")] = 25.0

    features = compute_indicators(prices)

    assert features[("NVDA", "rsi14")].iloc[14:].eq(100.0).all()
    assert features[("AAPL", "rsi14")].iloc[14:].eq(50.0).all()


def test_t02_all_features_are_causal_at_50_seeded_cut_dates():
    prices = _prices()
    full_stock = compute_indicators(prices)
    full_market = compute_market_features(prices)
    rng = np.random.default_rng(20261106)
    cut_points = rng.choice(np.arange(60, len(prices)), size=50, replace=False)

    for cut_point in cut_points:
        truncated_prices = prices.iloc[: cut_point + 1]
        truncated_stock = compute_indicators(truncated_prices)
        truncated_market = compute_market_features(truncated_prices)
        date = prices.index[cut_point]
        np.testing.assert_allclose(
            truncated_stock.loc[date].to_numpy(dtype=float),
            full_stock.loc[date].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            truncated_market.loc[date].to_numpy(dtype=float),
            full_market.loc[date].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )


def test_feature_pipeline_drops_shared_warmup_and_writes_metadata(tmp_path):
    input_path = tmp_path / "prices_clean.parquet"
    stock_path = tmp_path / "features_raw.parquet"
    market_path = tmp_path / "market_features_raw.parquet"
    metadata_path = tmp_path / "feature_meta.json"
    config_path = tmp_path / "config.yaml"
    prices = _prices(300)
    prices.to_parquet(input_path)
    config_path.write_text(
        "data:\n"
        f"  tickers: [{', '.join(TICKERS)}]\n"
        "features:\n"
        "  sma_short: 10\n"
        "  sma_long: 50\n"
        "  rsi: 14\n"
        "  vol: 20\n"
        "  volume: 20\n"
        "  market_corr: 20\n"
        "  market_dd: 252\n",
        encoding="utf-8",
    )

    stock, market = build_features(
        input_path, stock_path, market_path, metadata_path, config_path
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert stock.index.equals(market.index)
    assert stock.index[0] == prices.index[49]
    assert stock.notna().all().all() and market.notna().all().all()
    assert metadata["first_valid_date"] == prices.index[49].date().isoformat()
    assert pd.read_parquet(stock_path).equals(stock)
    assert pd.read_parquet(market_path).equals(market)
