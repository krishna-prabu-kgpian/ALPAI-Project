"""Align and validate cached adjusted OHLCV prices."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "prices_raw.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "prices_clean.parquet"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "data_quality.md"
PRICE_FIELDS = ("Open", "High", "Low", "Close")
ALL_FIELDS = (*PRICE_FIELDS, "Volume")
MAX_FILLED_FRACTION = 0.001
LARGE_MOVE_THRESHOLD = 0.25
LARGE_MOVE_EXPLANATIONS = {
    ("NVDA", "2016-11-11"): (
        "Post-earnings repricing following NVIDIA's Q3 FY2017 results, "
        "announced on 2016-11-10; the adjusted series shows a market move, "
        "not a split discontinuity."
    )
}


@dataclass(frozen=True)
class CleaningAudit:
    """Counts and returns needed for the data-quality report."""

    calendar_gaps_by_ticker: dict[str, int]
    filled_by_column: dict[str, int]
    original_missing_cells: int
    filled_cells: int
    total_cells: int
    largest_returns: dict[str, list[tuple[pd.Timestamp, float]]]

    @property
    def filled_fraction(self) -> float:
        return self.filled_cells / self.total_cells if self.total_cells else 0.0


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    try:
        tickers = config["data"]["tickers"]
        ffill_limit = config["data"]["ffill_limit"]
    except (TypeError, KeyError) as error:
        raise ValueError(f"Missing required data configuration in {config_path}") from error
    if not isinstance(tickers, list) or not tickers or not all(
        isinstance(ticker, str) and ticker for ticker in tickers
    ):
        raise ValueError("config data.tickers must be a non-empty list of symbols")
    if not isinstance(ffill_limit, int) or ffill_limit < 0:
        raise ValueError("config data.ffill_limit must be a non-negative integer")
    return config


def _validate_input(raw_prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if not isinstance(raw_prices, pd.DataFrame) or raw_prices.empty:
        raise ValueError("Raw prices must be a non-empty pandas DataFrame")
    if not isinstance(raw_prices.index, pd.DatetimeIndex):
        raise TypeError("Raw price index must be a DatetimeIndex")
    if raw_prices.index.hasnans:
        raise ValueError("Raw price index contains NaT values")
    if raw_prices.index.has_duplicates:
        raise ValueError("Raw price dates must be unique")
    if not raw_prices.index.is_monotonic_increasing:
        raise ValueError("Raw price dates must be strictly increasing")
    if not isinstance(raw_prices.columns, pd.MultiIndex) or raw_prices.columns.nlevels != 2:
        raise ValueError("Raw prices must have (ticker, field) MultiIndex columns")
    if raw_prices.columns.has_duplicates:
        raise ValueError("Raw prices contain duplicate ticker/field columns")

    expected_columns = pd.MultiIndex.from_product(
        [tickers, ALL_FIELDS], names=["ticker", "field"]
    )
    missing_columns = expected_columns.difference(raw_prices.columns)
    if len(missing_columns):
        raise ValueError(f"Raw prices are missing columns: {list(missing_columns)}")
    unexpected_tickers = set(raw_prices.columns.get_level_values(0)) - set(tickers)
    if unexpected_tickers:
        raise ValueError(f"Raw prices contain unexpected tickers: {sorted(unexpected_tickers)}")

    prices = raw_prices.reindex(columns=expected_columns)
    if prices.empty:
        raise ValueError("Raw prices have no dates")
    return prices


def _calendar_union(prices: pd.DataFrame, tickers: list[str]) -> pd.DatetimeIndex:
    calendars = []
    for ticker in tickers:
        ticker_prices = prices[ticker]
        valid_rows = ticker_prices.notna().any(axis=1)
        if not valid_rows.any():
            raise ValueError(f"Raw prices contain no data for ticker {ticker}")
        calendars.append(prices.index[valid_rows])
    union = prices.index
    for calendar in calendars[1:]:
        union = union.union(calendar)
    return union.sort_values()


def clean_prices(
    raw_prices: pd.DataFrame,
    tickers: list[str],
    ffill_limit: int = 5,
) -> tuple[pd.DataFrame, CleaningAudit]:
    """Align the ticker calendars, forward-fill prices only, and audit fills."""
    if ffill_limit < 0:
        raise ValueError("ffill_limit must be non-negative")
    prices = _validate_input(raw_prices, tickers)
    union_calendar = _calendar_union(prices, tickers)
    calendar_gaps_by_ticker = {
        ticker: len(union_calendar.difference(prices.index[prices[ticker].notna().any(axis=1)]))
        for ticker in tickers
    }
    prices = prices.reindex(union_calendar)

    original_missing_cells = int(prices.isna().sum().sum())
    filled_by_column: dict[str, int] = {}
    for ticker in tickers:
        for field in PRICE_FIELDS:
            column = (ticker, field)
            missing_before = prices[column].isna()
            prices[column] = prices[column].ffill(limit=ffill_limit)
            filled_by_column[f"{ticker}.{field}"] = int(
                (missing_before & prices[column].notna()).sum()
            )
        volume_column = (ticker, "Volume")
        missing_volume = prices[volume_column].isna()
        prices[volume_column] = prices[volume_column].fillna(0)
        filled_by_column[f"{ticker}.Volume"] = int(missing_volume.sum())

    remaining = prices.isna()
    if remaining.any().any():
        missing_columns = [
            f"{ticker}.{field}"
            for ticker, field in prices.columns[remaining.any(axis=0)]
        ]
        raise ValueError(
            "Price gaps remain after forward-fill (leading gaps cannot be filled "
            f"without look-ahead; check gaps beyond the configured limit): {missing_columns}"
        )

    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Cleaned OHLCV data contains NaN or infinite values")
    price_columns = [(ticker, field) for ticker in tickers for field in PRICE_FIELDS]
    if (prices.loc[:, price_columns] <= 0).any().any():
        raise ValueError("Cleaned OHLC prices must all be greater than zero")
    volume_columns = [(ticker, "Volume") for ticker in tickers]
    if (prices.loc[:, volume_columns] < 0).any().any():
        raise ValueError("Cleaned volume values must be non-negative")

    close = prices.loc[:, [(ticker, "Close") for ticker in tickers]]
    close.columns = tickers
    largest_returns: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for ticker in tickers:
        absolute_returns = close[ticker].pct_change().abs().dropna().nlargest(10)
        largest_returns[ticker] = [
            (pd.Timestamp(date), float(value))
            for date, value in absolute_returns.items()
        ]

    total_cells = int(prices.size)
    filled_cells = sum(filled_by_column.values())
    audit = CleaningAudit(
        calendar_gaps_by_ticker=calendar_gaps_by_ticker,
        filled_by_column=filled_by_column,
        original_missing_cells=original_missing_cells,
        filled_cells=filled_cells,
        total_cells=total_cells,
        largest_returns=largest_returns,
    )
    return prices, audit


def _validate_spec_row_count(prices: pd.DataFrame) -> None:
    if not 3580 <= len(prices) <= 3630:
        raise ValueError(
            f"Cleaned row count {len(prices)} is outside the spec range 3580–3630"
        )


def _format_report(prices: pd.DataFrame, audit: CleaningAudit, tickers: list[str]) -> str:
    report = [
        "# Price Data Quality Report",
        "",
        "## Dataset checks",
        "",
        f"- Rows: {len(prices)}",
        f"- Date range: {prices.index[0].date()} to {prices.index[-1].date()}",
        f"- Unique, strictly increasing dates: {'PASS' if prices.index.is_unique and prices.index.is_monotonic_increasing else 'FAIL'}",
        f"- Dates added when aligning ticker calendars: {sum(audit.calendar_gaps_by_ticker.values())}",
        f"- Non-finite values after cleaning: {int((~np.isfinite(prices.to_numpy(dtype=float))).sum())}",
        f"- Filled cells: {audit.filled_cells} / {audit.total_cells} "
        f"({audit.filled_fraction:.6%}; limit: {MAX_FILLED_FRACTION:.3%})",
        "",
        "## Filled cells by asset and field",
        "",
        "| Ticker | Open | High | Low | Close | Volume | Calendar gaps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ticker in tickers:
        counts = [
            audit.filled_by_column[f"{ticker}.{field}"] for field in ALL_FIELDS
        ]
        report.append(
            f"| {ticker} | "
            + " | ".join(str(count) for count in counts)
            + f" | {audit.calendar_gaps_by_ticker[ticker]} |"
        )

    report.extend(
        [
            "",
            "## Largest absolute single-day close returns",
            "",
            "Moves above 25% are integrity checks and must be explained before proceeding; "
            "this is not performance analysis.",
            "",
        ]
    )
    for ticker in tickers:
        report.extend(
            [
                f"### {ticker}",
                "",
                "| Date | Absolute daily return | Assessment |",
                "|---|---:|---|",
            ]
        )
        for date, value in audit.largest_returns[ticker]:
            assessment = "Within 25% threshold"
            if value > LARGE_MOVE_THRESHOLD:
                explanation = LARGE_MOVE_EXPLANATIONS.get(
                    (ticker, date.date().isoformat())
                )
                assessment = (
                    explanation
                    if explanation
                    else "**UNEXPLAINED — investigate before proceeding**"
                )
            report.append(f"| {date.date()} | {value:.2%} | {assessment} |")
        report.append("")
    return "\n".join(report).rstrip() + "\n"


def _atomic_write_parquet(prices: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        suffix=".parquet", dir=output_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        prices.to_parquet(temporary_path, index=True)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def clean_data(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """Clean the raw parquet, write the quality report, and save processed data."""
    config = load_config(config_path)
    tickers = config["data"]["tickers"]
    ffill_limit = config["data"]["ffill_limit"]
    raw_prices = pd.read_parquet(input_path)
    prices, audit = clean_prices(raw_prices, tickers, ffill_limit)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_format_report(prices, audit, tickers), encoding="utf-8")

    _validate_spec_row_count(prices)
    if audit.filled_fraction > MAX_FILLED_FRACTION:
        raise ValueError(
            f"Filled {audit.filled_cells}/{audit.total_cells} cells "
            f"({audit.filled_fraction:.3%}), exceeding the 0.1% investigation threshold; "
            f"review {report_path}"
        )

    unexplained_moves = [
        (ticker, date.date().isoformat(), value)
        for ticker, returns in audit.largest_returns.items()
        for date, value in returns
        if value > LARGE_MOVE_THRESHOLD
        and (ticker, date.date().isoformat()) not in LARGE_MOVE_EXPLANATIONS
    ]
    if unexplained_moves:
        raise ValueError(
            "Daily close moves above 25% need explanation before proceeding: "
            f"{unexplained_moves}; see {report_path}"
        )

    _atomic_write_parquet(prices, output_path)
    return prices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    prices = clean_data(args.input, args.output, args.report, args.config)
    print(
        f"Cleaned {len(prices)} rows ({prices.index[0].date()} to "
        f"{prices.index[-1].date()}) to {args.output}"
    )
    print(f"Data-quality report: {args.report}")


if __name__ == "__main__":
    main()
