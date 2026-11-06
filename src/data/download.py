"""Download and cache adjusted daily OHLCV prices."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "prices_raw.parquet"
PRICE_FIELDS = ("Open", "High", "Low", "Close", "Volume")
FIELD_LOOKUP = {field.lower(): field for field in PRICE_FIELDS}


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load configuration and validate the inputs required for ingestion."""
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    try:
        data_config = config["data"]
        tickers = data_config["tickers"]
        start = data_config["start"]
        end = data_config["end"]
        auto_adjust = data_config["auto_adjust"]
    except (TypeError, KeyError) as error:
        raise ValueError(f"Missing required data configuration in {config_path}") from error

    if not isinstance(tickers, list) or not tickers or not all(
        isinstance(ticker, str) and ticker for ticker in tickers
    ):
        raise ValueError("config data.tickers must be a non-empty list of symbols")
    if not isinstance(start, str) or not isinstance(end, str) or start >= end:
        raise ValueError("config data.start and data.end must be ordered ISO date strings")
    if auto_adjust is not True:
        raise ValueError("Price ingestion requires data.auto_adjust: true (spec D-02)")
    return config


def normalize_columns(frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance's supported column layouts to (ticker, field)."""
    if not isinstance(frame.columns, pd.MultiIndex):
        if len(tickers) != 1:
            raise ValueError("Expected MultiIndex columns for a multi-ticker download")
        columns = frame.copy()
        columns.columns = pd.MultiIndex.from_product(
            [tickers, [FIELD_LOOKUP.get(str(column).lower(), str(column)) for column in columns.columns]],
            names=["ticker", "field"],
        )
        return _validate_normalized_columns(columns, tickers)

    normalized_columns: list[tuple[str, str]] = []
    keep_positions: list[int] = []
    for position, column in enumerate(frame.columns):
        if len(column) != 2:
            raise ValueError(f"Unexpected yfinance column: {column!r}")
        first, second = (str(value) for value in column)
        if first.lower() == "adj close" or second.lower() == "adj close":
            continue
        if first.lower() in FIELD_LOOKUP and second in tickers:
            normalized_columns.append((second, FIELD_LOOKUP[first.lower()]))
        elif second.lower() in FIELD_LOOKUP and first in tickers:
            normalized_columns.append((first, FIELD_LOOKUP[second.lower()]))
        else:
            raise ValueError(f"Unexpected yfinance column: {column!r}")
        keep_positions.append(position)

    normalized = frame.iloc[:, keep_positions].copy()
    normalized.columns = pd.MultiIndex.from_tuples(
        normalized_columns, names=["ticker", "field"]
    )
    return _validate_normalized_columns(normalized, tickers)


def _validate_normalized_columns(frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    expected = pd.MultiIndex.from_product(
        [tickers, PRICE_FIELDS], names=["ticker", "field"]
    )
    missing = expected.difference(frame.columns)
    if len(missing):
        raise ValueError(f"Download is missing required OHLCV columns: {list(missing)}")
    if frame.columns.has_duplicates:
        raise ValueError("Download contains duplicate ticker/field columns")
    return frame.reindex(columns=expected).sort_index()


def _download_from_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as error:
        raise RuntimeError(
            "yfinance is required to fetch prices; install dependencies from requirements.txt"
        ) from error

    return yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(".meta.json")


def _library_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in ("yfinance", "pandas", "numpy")
    }


def _read_cache(output_path: Path, tickers: list[str]) -> pd.DataFrame:
    metadata_path = _metadata_path(output_path)
    if not output_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Price cache is incomplete: expected both {output_path} and {metadata_path}"
        )

    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if metadata.get("sha256") != _sha256(output_path):
        raise ValueError(f"Price cache hash does not match metadata: {output_path}")

    prices = pd.read_parquet(output_path)
    if metadata.get("row_count") != len(prices):
        raise ValueError(f"Price cache row count does not match metadata: {output_path}")
    if not isinstance(prices.columns, pd.MultiIndex):
        raise ValueError("Cached prices must have (ticker, field) MultiIndex columns")
    return _validate_normalized_columns(prices, tickers)


def _write_cache(prices: pd.DataFrame, output_path: Path, config: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = _metadata_path(output_path)

    parquet_fd, parquet_temp_name = tempfile.mkstemp(
        suffix=".parquet", dir=output_path.parent
    )
    os.close(parquet_fd)
    metadata_fd, metadata_temp_name = tempfile.mkstemp(
        suffix=".json", dir=output_path.parent
    )
    os.close(metadata_fd)
    parquet_temp = Path(parquet_temp_name)
    metadata_temp = Path(metadata_temp_name)

    try:
        prices.to_parquet(parquet_temp, index=True)
        metadata = {
            "sha256": _sha256(parquet_temp),
            "row_count": len(prices),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "versions": _library_versions(),
            "tickers": config["data"]["tickers"],
            "start": config["data"]["start"],
            "end": config["data"]["end"],
            "auto_adjust": config["data"]["auto_adjust"],
            "columns": [list(column) for column in prices.columns],
        }
        with metadata_temp.open("w", encoding="utf-8", newline="\n") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
            metadata_file.write("\n")
        os.replace(parquet_temp, output_path)
        os.replace(metadata_temp, metadata_path)
    finally:
        parquet_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)


def _validate_download(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("yfinance returned no price rows")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("yfinance result must have a DatetimeIndex")

    prices = prices.sort_index()
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and strictly increasing")

    if (
        config["data"]["start"] == "2012-01-01"
        and config["data"]["end"] == "2026-05-01"
    ):
        if prices.index[0].date().isoformat() != "2012-01-03":
            raise ValueError(
                f"Expected first price date 2012-01-03, got {prices.index[0].date()}"
            )
        if prices.index[-1].date().isoformat() != "2026-04-30":
            raise ValueError(
                f"Expected final price date 2026-04-30, got {prices.index[-1].date()}"
            )
        if not 3580 <= len(prices) <= 3630:
            raise ValueError(
                f"Unexpected price row count {len(prices)}; expected 3580–3630"
            )

    close = prices.xs("Close", axis=1, level="field")
    if close.isna().all(axis=0).any():
        missing_tickers = list(close.columns[close.isna().all(axis=0)])
        raise ValueError(f"No adjusted close data returned for: {missing_tickers}")
    if not np.isfinite(close.to_numpy(dtype=float)).any():
        raise ValueError("yfinance returned no finite adjusted close values")
    return prices


def download_prices(
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    refresh: bool = False,
    downloader: Callable[[list[str], str, str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Load the offline cache or fetch and persist prices when it is absent."""
    config = load_config(config_path)
    data_config = config["data"]
    output_path = Path(output_path)

    if output_path.exists() and not refresh:
        return _read_cache(output_path, data_config["tickers"])
    if not refresh and _metadata_path(output_path).exists():
        raise FileNotFoundError(
            f"Price metadata exists without its parquet cache: {output_path}"
        )

    fetch = downloader or _download_from_yfinance
    raw_prices = fetch(
        data_config["tickers"], data_config["start"], data_config["end"]
    )
    if not isinstance(raw_prices, pd.DataFrame):
        raise TypeError("The price downloader must return a pandas DataFrame")
    if raw_prices.empty:
        raise ValueError(
            "yfinance returned no price rows; check the network and Yahoo Finance response"
        )
    prices = normalize_columns(raw_prices, data_config["tickers"])
    prices = _validate_download(prices, config)
    _write_cache(prices, output_path, config)
    return prices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="download again instead of using the cache; first record the change in the Decision Log",
    )
    args = parser.parse_args()

    prices = download_prices(args.config, args.output, refresh=args.refresh)
    print(f"Cached {len(prices)} rows at {args.output}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")


if __name__ == "__main__":
    main()
