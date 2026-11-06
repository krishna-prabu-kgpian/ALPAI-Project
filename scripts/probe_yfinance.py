"""Inspect yfinance's multi-ticker column layout using a short sample period."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

try:
    import yfinance as yf
except ImportError as error:
    raise SystemExit("Install yfinance from requirements.txt before running this probe") from error

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml"
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as config_file:
        tickers = yaml.safe_load(config_file)["data"]["tickers"]
    sample = yf.download(
        tickers,
        start="2024-01-02",
        end="2024-01-09",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    print(f"Shape: {sample.shape}")
    print(f"Index type: {type(sample.index).__name__}")
    print(f"Columns type: {type(sample.columns).__name__}")
    print(f"Column level names: {sample.columns.names}")
    print(f"First columns: {list(sample.columns[:10])}")
    print(f"MultiIndex columns: {isinstance(sample.columns, pd.MultiIndex)}")


if __name__ == "__main__":
    main()
