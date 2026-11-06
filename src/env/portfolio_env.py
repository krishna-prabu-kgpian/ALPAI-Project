"""Discrete portfolio actions and observation-vector construction."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

N_BUCKETS = 6
N_ACTIONS = 31
N_STOCK_FEATURES = 30
N_REGIME_PROBABILITIES = 3
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTION_TABLE_PATH = PROJECT_ROOT / "artifacts" / "action_table.json"

Action: TypeAlias = dict[str, int | None]


def build_action_table() -> list[Action]:
    """Return HOLD followed by all ordered, non-self bucket transfers."""
    actions: list[Action] = [{"action_id": 0, "from": None, "to": None}]
    for source in range(N_BUCKETS):
        for destination in range(N_BUCKETS):
            if source != destination:
                actions.append(
                    {
                        "action_id": len(actions),
                        "from": source,
                        "to": destination,
                    }
                )
    return actions


_ACTION_TABLE = tuple(build_action_table())


def decode_action(action_id: int) -> Action:
    """Look up one action by its stable discrete action ID."""
    if isinstance(action_id, bool) or not isinstance(action_id, int):
        raise TypeError("action_id must be an integer")
    if not 0 <= action_id < N_ACTIONS:
        raise ValueError(f"action_id must be between 0 and {N_ACTIONS - 1}")
    return dict(_ACTION_TABLE[action_id])


def save_action_table(path: Path = DEFAULT_ACTION_TABLE_PATH) -> None:
    """Write the canonical action lookup table as formatted JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".json", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as table_file:
            json.dump(_ACTION_TABLE, table_file, indent=2)
            table_file.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_observation(
    stock_features: ArrayLike,
    portfolio_weights: ArrayLike,
    regime_probabilities: ArrayLike | None,
    *,
    include_weights_in_obs: bool = True,
    regime_enabled: bool = True,
    include_novelty: bool = False,
    novelty: float | None = None,
) -> NDArray[np.float32]:
    """Build the configured float32 observation in the E-04 feature order.

    ``stock_features`` must already be flattened asset-major in F-06 order.
    Regime probabilities must be the continuous-series, EMA-smoothed values
    produced by the regime model.
    """
    features = np.asarray(stock_features, dtype=float)
    if features.shape != (N_STOCK_FEATURES,) or not np.isfinite(features).all():
        raise ValueError("stock_features must contain 30 finite values in F-06 order")

    parts = [features]
    if include_weights_in_obs:
        weights = np.asarray(portfolio_weights, dtype=float)
        if (
            weights.shape != (N_BUCKETS,)
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or abs(float(weights.sum()) - 1.0) >= 1e-9
        ):
            raise ValueError("portfolio_weights must be six valid weights summing to 1")
        parts.append(weights)

    if regime_enabled:
        if regime_probabilities is None:
            raise ValueError("regime_probabilities are required when regime is enabled")
        probabilities = np.asarray(regime_probabilities, dtype=float)
        if (
            probabilities.shape != (N_REGIME_PROBABILITIES,)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or abs(float(probabilities.sum()) - 1.0) >= 1e-9
        ):
            raise ValueError(
                "regime_probabilities must be three valid probabilities summing to 1"
            )
        parts.append(probabilities)
        if include_novelty:
            if novelty is None or not np.isfinite(novelty):
                raise ValueError("a finite novelty score is required when enabled")
            parts.append(np.array([novelty], dtype=float))

    observation = np.concatenate(parts).astype(np.float32)
    if not np.isfinite(observation).all():
        raise ValueError("observation is not finite after float32 conversion")
    return observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ACTION_TABLE_PATH)
    args = parser.parse_args()
    save_action_table(args.output)
    print(f"Saved {N_ACTIONS}-action table to {args.output}")


if __name__ == "__main__":
    main()
