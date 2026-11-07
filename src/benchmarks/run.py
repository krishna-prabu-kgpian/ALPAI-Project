"""Run the three benchmarks over a scored window and write C-01 CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.agents.curriculum import load_config
from src.benchmarks.buy_hold import run_buy_hold
from src.benchmarks.equal_weight import run_equal_weight
from src.benchmarks.mvo import run_mvo
from src.benchmarks.simulator import load_market_data
from src.evaluation.backtest import finalize_frame, require_freeze
from src.evaluation.metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGIME_LABELS = {"val_2022": "val_2022", "test_normal": "normal"}


def run_benchmarks(window: str, output_dir: Path) -> pd.DataFrame:
    """Write one CSV per benchmark and return their window metrics."""
    require_freeze(window)
    config = load_config()
    market = load_market_data("tuning" if window == "val_2022" else "final")
    first, last = market.window_bounds(window, config)
    settings = config["benchmarks"]
    frames = {
        "buy_hold": run_buy_hold(market, first, last),
        "equal_weight": run_equal_weight(
            market, first, last, rebalance_every=int(settings["rebalance_every"])
        ),
        "mvo": run_mvo(
            market,
            first,
            last,
            window=int(settings["mvo_window"]),
            rebalance_every=int(settings["rebalance_every"]),
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for strategy, frame in frames.items():
        finalize_frame(frame, REGIME_LABELS.get(window, window)).to_csv(
            output_dir / f"{strategy}.csv", index=False
        )
        metrics = compute_metrics(frame["daily_return"], frame["turnover"], frame["cost_paid"])
        summary.append({"strategy": strategy, "scenario": window, **metrics})
    return pd.DataFrame(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="val_2022")
    args = parser.parse_args()
    folder = "validation" if args.window == "val_2022" else "backtests"
    output_dir = PROJECT_ROOT / "results" / folder / args.window
    summary = run_benchmarks(args.window, output_dir)
    summary.to_csv(output_dir / "benchmark_metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
