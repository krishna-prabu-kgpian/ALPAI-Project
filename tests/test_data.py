import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_data, clean_prices


TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def _valid_prices(rows=8):
    dates = pd.date_range("2020-01-02", periods=rows, freq="B")
    columns = pd.MultiIndex.from_product(
        [TICKERS, FIELDS], names=["ticker", "field"]
    )
    return pd.DataFrame(10.0, index=dates, columns=columns)


def _write_config(path, ffill_limit=5):
    path.write_text(
        "data:\n"
        f"  tickers: [{', '.join(TICKERS)}]\n"
        f"  ffill_limit: {ffill_limit}\n",
        encoding="utf-8",
    )


def _full_length_prices(rows=3602):
    dates = pd.date_range("2012-01-03", periods=rows, freq="B")
    columns = pd.MultiIndex.from_product(
        [TICKERS, FIELDS], names=["ticker", "field"]
    )
    return pd.DataFrame(10.0, index=dates, columns=columns)


def test_t01_cleaned_prices_has_valid_values_and_dates():
    raw_prices = _valid_prices()
    raw_prices[("AAA", "Close")] = np.linspace(10.0, 10.7, len(raw_prices))

    cleaned, _ = clean_prices(raw_prices, TICKERS)

    values = cleaned.to_numpy(dtype=float)
    close_columns = [(ticker, field) for ticker in TICKERS for field in FIELDS[:4]]
    assert np.isfinite(values).all()
    assert cleaned.index.is_unique
    assert cleaned.index.is_monotonic_increasing
    assert (cleaned.loc[:, close_columns] > 0).all().all()


def test_forward_fills_prices_but_fills_missing_volume_with_zero():
    raw_prices = _valid_prices()
    raw_prices.loc[raw_prices.index[2], ("AAA", "Close")] = np.nan
    raw_prices.loc[raw_prices.index[3], ("AAA", "Volume")] = np.nan

    cleaned, audit = clean_prices(raw_prices, TICKERS, ffill_limit=5)

    assert cleaned.loc[cleaned.index[2], ("AAA", "Close")] == 10.0
    assert cleaned.loc[cleaned.index[3], ("AAA", "Volume")] == 0.0
    assert audit.filled_by_column["AAA.Close"] == 1
    assert audit.filled_by_column["AAA.Volume"] == 1


def test_leading_price_gap_fails_instead_of_looking_ahead():
    raw_prices = _valid_prices()
    raw_prices.loc[raw_prices.index[0], ("AAA", "Close")] = np.nan

    with pytest.raises(ValueError, match="leading gaps cannot be filled"):
        clean_prices(raw_prices, TICKERS)


def test_price_gaps_beyond_ffill_limit_fail():
    raw_prices = _valid_prices()
    raw_prices.loc[raw_prices.index[:2], ("AAA", "Close")] = np.nan

    with pytest.raises(ValueError, match="gaps beyond the configured limit"):
        clean_prices(raw_prices, TICKERS, ffill_limit=1)


def test_unsorted_or_duplicate_dates_fail():
    raw_prices = _valid_prices()
    unsorted = raw_prices.iloc[::-1]
    duplicate = pd.concat([raw_prices, raw_prices.iloc[[-1]]])

    with pytest.raises(ValueError, match="strictly increasing"):
        clean_prices(unsorted, TICKERS)
    with pytest.raises(ValueError, match="must be unique"):
        clean_prices(duplicate, TICKERS)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (("AAA", "Close"), 0.0, "greater than zero"),
        (("BBB", "Volume"), -1.0, "non-negative"),
        (("CCC", "Open"), np.inf, "infinite values"),
    ],
)
def test_invalid_price_or_volume_fails(column, value, message):
    raw_prices = _valid_prices()
    raw_prices.loc[raw_prices.index[0], column] = value

    with pytest.raises(ValueError, match=message):
        clean_prices(raw_prices, TICKERS)


def test_clean_data_writes_parquet_and_quality_report(tmp_path):
    raw_path = tmp_path / "prices_raw.parquet"
    output_path = tmp_path / "prices_clean.parquet"
    report_path = tmp_path / "data_quality.md"
    config_path = tmp_path / "config.yaml"
    _full_length_prices().to_parquet(raw_path)
    _write_config(config_path)

    cleaned = clean_data(raw_path, output_path, report_path, config_path)
    saved = pd.read_parquet(output_path)
    report = report_path.read_text(encoding="utf-8")

    pd.testing.assert_frame_equal(cleaned, saved)
    assert "Filled cells: 0 / 90050" in report
    assert "Largest absolute single-day close returns" in report


def test_fill_threshold_writes_diagnostic_and_blocks_processed_output(tmp_path):
    raw_path = tmp_path / "prices_raw.parquet"
    output_path = tmp_path / "prices_clean.parquet"
    report_path = tmp_path / "data_quality.md"
    config_path = tmp_path / "config.yaml"
    prices = _full_length_prices()
    prices.loc[prices.index[1::2][:91], ("AAA", "Close")] = np.nan
    prices.to_parquet(raw_path)
    _write_config(config_path)

    with pytest.raises(ValueError, match="exceeding the 0.1% investigation threshold"):
        clean_data(raw_path, output_path, report_path, config_path)

    assert report_path.exists()
    assert not output_path.exists()


def test_unexplained_large_move_is_reported_and_blocks_processed_output(tmp_path):
    raw_path = tmp_path / "prices_raw.parquet"
    output_path = tmp_path / "prices_clean.parquet"
    report_path = tmp_path / "data_quality.md"
    config_path = tmp_path / "config.yaml"
    prices = _full_length_prices()
    prices.loc[prices.index[1], ("AAA", "Close")] = 13.0
    prices.to_parquet(raw_path)
    _write_config(config_path)

    with pytest.raises(ValueError, match="need explanation before proceeding"):
        clean_data(raw_path, output_path, report_path, config_path)

    assert "UNEXPLAINED — investigate before proceeding" in report_path.read_text(
        encoding="utf-8"
    )
    assert not output_path.exists()
