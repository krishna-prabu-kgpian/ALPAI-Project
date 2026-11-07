"""Compressed-state tabular Q-learning (M-02): 3 x 3 x 5 x 3 = 135 states."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
N_ACTIONS = 31
N_STATES = 3 * 3 * 5 * 3
N_ASSETS = 5
FEATURES_PER_ASSET = 6
SMA50_GAP_INDEX = 2
WEIGHTS_SLICE = slice(30, 36)
REGIME_SLICE = slice(36, 39)
CASH_EDGES = (0.05, 0.5)
FIT_LABELS = {"final": "2012-2023", "tuning": "2012-2021"}


def sma50_columns() -> list[int]:
    """Observation positions of each asset's z-scored sma50_gap (F-06 asset-major)."""
    return [asset * FEATURES_PER_ASSET + SMA50_GAP_INDEX for asset in range(N_ASSETS)]


def fit_tabular_cuts(fit: str = "final") -> dict[str, Any]:
    """Fit trend terciles on the fit window only and keep the sma50_gap z-score inverse."""
    label = FIT_LABELS[fit]
    processed = PROJECT_ROOT / "data" / "processed"
    raw = pd.read_parquet(processed / "features_raw.parquet")
    scaler = json.loads(
        (PROJECT_ROOT / "artifacts" / f"scaler_{label}.json").read_text(encoding="utf-8")
    )
    start = pd.Timestamp(scaler["fit_window"]["start"])
    end = pd.Timestamp(scaler["fit_window"]["end"])
    gap = raw.xs("sma50_gap", axis=1, level=1).loc[(raw.index >= start) & (raw.index <= end)]
    trend = gap.mean(axis=1).to_numpy()
    lower, upper = np.quantile(trend, [1 / 3, 2 / 3])
    columns = sma50_columns()
    return {
        "fit_window": label,
        "trend_terciles": [float(lower), float(upper)],
        "cash_edges": list(CASH_EDGES),
        "sma50_mean": [float(scaler["means"][i]) for i in columns],
        "sma50_std": [
            float(max(scaler["standard_deviations"][i], scaler["standard_deviation_floor"]))
            for i in columns
        ],
    }


def save_cuts(cuts: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".json", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as cuts_file:
            json.dump(cuts, cuts_file, indent=2)
            cuts_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def state_components(observation: NDArray[np.float32], cuts: dict[str, Any]) -> tuple[int, int, int, int]:
    """(market_regime, trend_regime, leader, cash_bucket) of a 39-D observation."""
    obs = np.asarray(observation, dtype=np.float64)
    gap = obs[sma50_columns()] * np.asarray(cuts["sma50_std"]) + np.asarray(cuts["sma50_mean"])
    market_regime = int(np.argmax(obs[REGIME_SLICE]))
    trend_regime = int(np.searchsorted(cuts["trend_terciles"], gap.mean(), side="right"))
    leader = int(np.argmax(gap))
    cash_bucket = int(np.searchsorted(cuts["cash_edges"], obs[WEIGHTS_SLICE][5], side="right"))
    return market_regime, trend_regime, leader, cash_bucket


def state_index(observation: NDArray[np.float32], cuts: dict[str, Any]) -> int:
    """Pure, deterministic mapping of an observation to one of 135 states."""
    regime, trend, leader, cash = state_components(observation, cuts)
    return ((regime * 3 + trend) * 5 + leader) * 3 + cash


class TabularQAgent:
    """Zero-initialised Q-table with one-step Q-learning updates."""

    kind = "tabular_q"

    def __init__(
        self,
        cuts: dict[str, Any],
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
        seed: int | None = None,
    ) -> None:
        self.cuts = cuts
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.q_table = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self.visits = np.zeros(N_STATES, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def greedy_action(self, observation: NDArray[np.float32]) -> int:
        return int(np.argmax(self.q_table[state_index(observation, self.cuts)]))

    def act(self, observation: NDArray[np.float32], epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(N_ACTIONS))
        return self.greedy_action(observation)

    def observe(
        self,
        observation: NDArray[np.float32],
        action: int,
        reward: float,
        next_observation: NDArray[np.float32],
        done: bool,
    ) -> float:
        state = state_index(observation, self.cuts)
        next_state = state_index(next_observation, self.cuts)
        target = reward + (0.0 if done else self.gamma * self.q_table[next_state].max())
        error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * error
        self.visits[state] += 1
        return float(error**2)

    def state_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "cuts": self.cuts, "q_table": self.q_table, "visits": self.visits}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.cuts = state["cuts"]
        self.q_table = np.asarray(state["q_table"], dtype=np.float64)
        self.visits = np.asarray(state["visits"], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and save tabular cut-points.")
    parser.add_argument("--fit", choices=sorted(FIT_LABELS), default="final")
    args = parser.parse_args()
    suffix = "" if args.fit == "final" else f"_{FIT_LABELS[args.fit]}"
    path = PROJECT_ROOT / "artifacts" / f"tabular_cuts{suffix}.json"
    save_cuts(fit_tabular_cuts(args.fit), path)
    print(f"Saved tabular cut-points to {path}")


if __name__ == "__main__":
    main()
