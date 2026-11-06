"""Build causal raw stock and market features from cleaned prices."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.features.indicators import FEATURE_ORDER, compute_indicators
from src.features.market_features import (
    MARKET_FEATURE_ORDER,
    compute_market_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "prices_clean.parquet"
DEFAULT_STOCK_OUTPUT = PROJECT_ROOT / "data" / "processed" / "features_raw.parquet"
DEFAULT_MARKET_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "market_features_raw.parquet"
)
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "feature_meta.json"


def build_features(
    input_path: Path = DEFAULT_INPUT_PATH,
    stock_output_path: Path = DEFAULT_STOCK_OUTPUT,
    market_output_path: Path = DEFAULT_MARKET_OUTPUT,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build continuous-series features and save shared warm-up-trimmed outputs."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    try:
        data_config = config["data"]
        feature_config = config["features"]
        tickers = data_config["tickers"]
        sma_short = feature_config["sma_short"]
        sma_long = feature_config["sma_long"]
        rsi_period = feature_config["rsi"]
        vol_window = feature_config["vol"]
        volume_window = feature_config["volume"]
        corr_window = feature_config["market_corr"]
        drawdown_window = feature_config["market_dd"]
    except (TypeError, KeyError) as error:
        raise ValueError(f"Missing feature configuration in {config_path}") from error

    prices = pd.read_parquet(input_path)
    _validate_ticker_order(prices, tickers)
    stock_features = compute_indicators(
        prices,
        sma_short=sma_short,
        sma_long=sma_long,
        rsi_period=rsi_period,
        vol_window=vol_window,
        volume_window=volume_window,
    )
    market_features = compute_market_features(
        prices,
        vol_window=vol_window,
        corr_window=corr_window,
        drawdown_window=drawdown_window,
    )

    valid_rows = stock_features.notna().all(axis=1) & market_features.notna().all(axis=1)
    if not valid_rows.any():
        raise ValueError("No dates have complete stock and market features")
    first_valid_date = pd.Timestamp(valid_rows[valid_rows].index[0])
    stock_features = stock_features.loc[valid_rows]
    market_features = market_features.loc[valid_rows]
    if not stock_features.index.equals(market_features.index):
        raise RuntimeError("Stock and market feature dates are not aligned")
    if stock_features.isna().any().any() or market_features.isna().any().any():
        raise RuntimeError("Warm-up-trimmed features still contain missing values")

    _atomic_write_parquet(stock_features, Path(stock_output_path))
    _atomic_write_parquet(market_features, Path(market_output_path))
    metadata = {
        "source": str(Path(input_path).resolve()),
        "source_rows": len(prices),
        "rows_after_warmup": len(stock_features),
        "warmup_rows_dropped": len(prices) - len(stock_features),
        "first_valid_date": first_valid_date.date().isoformat(),
        "last_valid_date": stock_features.index[-1].date().isoformat(),
        "tickers": list(tickers),
        "stock_feature_order": list(FEATURE_ORDER),
        "market_feature_order": list(MARKET_FEATURE_ORDER),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(metadata, Path(metadata_path))
    return stock_features, market_features


def _validate_ticker_order(prices: pd.DataFrame, tickers: list[str]) -> None:
    if not isinstance(prices.columns, pd.MultiIndex) or prices.columns.nlevels != 2:
        raise ValueError("Cleaned prices must have (ticker, field) MultiIndex columns")
    present_tickers = list(dict.fromkeys(prices.columns.get_level_values(0)))
    if present_tickers != list(tickers):
        raise ValueError(
            "Cleaned price ticker order must match config data.tickers: "
            f"expected {list(tickers)}, found {present_tickers}"
        )


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".parquet", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path, index=True)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".json", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as metadata_file:
            json.dump(value, metadata_file, indent=2)
            metadata_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--stock-output", type=Path, default=DEFAULT_STOCK_OUTPUT)
    parser.add_argument("--market-output", type=Path, default=DEFAULT_MARKET_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    args = parser.parse_args()

    stock, market = build_features(
        args.input,
        args.stock_output,
        args.market_output,
        args.metadata,
        args.config,
    )
    print(
        f"Built {len(stock)} rows from {stock.index[0].date()} to "
        f"{stock.index[-1].date()}"
    )
    print(f"Stock features: {args.stock_output}")
    print(f"Market features: {args.market_output}")
    print(f"Feature metadata: {args.metadata}")


if __name__ == "__main__":
    main()
