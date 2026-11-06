import json

import pandas as pd
import pytest

from src.data.download import download_prices, normalize_columns


TICKERS = ["AAA", "BBB"]
FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def _sample_frame(field_first=True):
    dates = pd.date_range("2020-01-02", periods=3, freq="B")
    if field_first:
        columns = pd.MultiIndex.from_product([FIELDS, TICKERS])
        values = [[float(i + 1) for i in range(len(columns))] for _ in dates]
    else:
        columns = pd.MultiIndex.from_product([TICKERS, FIELDS])
        values = [[float(i + 1) for i in range(len(columns))] for _ in dates]
    return pd.DataFrame(values, index=dates, columns=columns)


@pytest.mark.parametrize("field_first", [True, False])
def test_normalize_yfinance_column_orders(field_first):
    prices = normalize_columns(_sample_frame(field_first), TICKERS)

    assert prices.columns.names == ["ticker", "field"]
    assert list(prices.columns) == [
        (ticker, field) for ticker in TICKERS for field in FIELDS
    ]


def test_normalize_ignores_adjusted_close_column():
    frame = _sample_frame()
    frame[("Adj Close", "AAA")] = 1.0
    prices = normalize_columns(frame, TICKERS)

    assert len(prices.columns) == len(TICKERS) * len(FIELDS)


def test_default_load_uses_verified_cache_without_downloading(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'data:\n  tickers: [AAA, BBB]\n  start: "2020-01-01"\n'
        '  end: "2020-01-10"\n  auto_adjust: true\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "prices_raw.parquet"
    returned = _sample_frame()

    def downloader(tickers, start, end):
        assert tickers == TICKERS
        assert start == "2020-01-01"
        assert end == "2020-01-10"
        return returned

    # Keep the focused test independent of the installed yfinance distribution.
    import src.data.download as download_module

    monkeypatch.setattr(
        download_module,
        "_library_versions",
        lambda: {"yfinance": "test", "pandas": pd.__version__, "numpy": "test"},
    )
    saved = download_prices(config_path, output_path, downloader=downloader)

    def unexpected_download(*args):
        raise AssertionError("A valid cache must not trigger a network request")

    loaded = download_prices(config_path, output_path, downloader=unexpected_download)
    metadata = json.loads(output_path.with_suffix(".meta.json").read_text(encoding="utf-8"))

    pd.testing.assert_frame_equal(saved, loaded, check_freq=False)
    assert metadata["row_count"] == 3
    assert metadata["sha256"]
    assert metadata["timestamp_utc"].endswith("+00:00")


def test_cache_hash_mismatch_is_reported(tmp_path):
    output_path = tmp_path / "prices_raw.parquet"
    output_path.touch()
    output_path.with_suffix(".meta.json").write_text('{"sha256":"wrong"}', encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'data:\n  tickers: [AAA, BBB]\n  start: "2020-01-01"\n'
        '  end: "2020-01-10"\n  auto_adjust: true\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash does not match"):
        download_prices(config_path, output_path)


def test_empty_download_is_reported_without_creating_cache(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'data:\n  tickers: [AAA, BBB]\n  start: "2020-01-01"\n'
        '  end: "2020-01-10"\n  auto_adjust: true\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "prices_raw.parquet"

    with pytest.raises(ValueError, match="returned no price rows"):
        download_prices(
            config_path,
            output_path,
            downloader=lambda *_: pd.DataFrame(),
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".meta.json").exists()


def test_refresh_bypasses_existing_cache(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        'data:\n  tickers: [AAA, BBB]\n  start: "2020-01-01"\n'
        '  end: "2020-01-10"\n  auto_adjust: true\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "prices_raw.parquet"
    import src.data.download as download_module

    monkeypatch.setattr(
        download_module,
        "_library_versions",
        lambda: {"yfinance": "test", "pandas": pd.__version__, "numpy": "test"},
    )
    download_prices(
        config_path,
        output_path,
        downloader=lambda *_: _sample_frame(),
    )
    refresh_calls = []

    def downloader(*args):
        refresh_calls.append(args)
        return _sample_frame()

    download_prices(config_path, output_path, refresh=True, downloader=downloader)

    assert len(refresh_calls) == 1
