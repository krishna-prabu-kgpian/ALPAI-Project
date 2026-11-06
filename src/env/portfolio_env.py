"""Discrete portfolio actions, observation-vector construction, and Gymnasium env."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypeAlias

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray

from src.env.costs import CostModel
from src.env.portfolio import mark_to_market, apply_trades
from src.env.reward import calculate_reward

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
    """Build the configured float32 observation in the E-04 feature order."""
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


class PortfolioEnv(gym.Env):
    """Gymnasium portfolio environment for a five-asset, one-cash allocation."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: ArrayLike | str | Path | None = None,
        regime_probabilities: ArrayLike | str | Path | None = None,
        *,
        asset_returns: ArrayLike | None = None,
        start_idx: int = 0,
        length: int = 126,
        initial_value: float = 1.0,
        initial_weights: ArrayLike | None = None,
        include_weights_in_obs: bool = True,
        regime_enabled: bool = True,
        include_novelty: bool = False,
        cost: float = 0.001,
        rebalance_step: float = 0.10,
        lambda1: float | None = None,
        lambda2: float | None = None,
        reward_scale: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if initial_weights is None:
            initial_weights = np.array([0.2] * 5 + [0.0], dtype=np.float64)
        self.initial_value = float(initial_value)
        self.initial_weights = np.asarray(initial_weights, dtype=np.float64)
        if (
            self.initial_weights.shape != (N_BUCKETS,)
            or not np.isfinite(self.initial_weights).all()
            or (self.initial_weights < 0).any()
            or abs(float(self.initial_weights.sum()) - 1.0) >= 1e-9
        ):
            raise ValueError("initial_weights must be six non-negative weights summing to 1")
        if not np.isfinite(self.initial_value) or self.initial_value <= 0:
            raise ValueError("initial_value must be finite and positive")
        self.include_weights_in_obs = include_weights_in_obs
        self.regime_enabled = regime_enabled
        self.include_novelty = include_novelty
        self.length = int(length)
        self.start_idx = int(start_idx)
        self.rebalance_step = float(rebalance_step)
        if not 0 < self.rebalance_step <= 1:
            raise ValueError("rebalance_step must be in (0, 1]")
        self.cost_model = CostModel(cost_rate=float(cost), slippage_mode="none")
        config = self._load_config()
        reward_config = config.get("reward", {})
        self.lambda1 = self._coalesce_reward_value(lambda1, reward_config.get("lambda1"))
        self.lambda2 = self._coalesce_reward_value(lambda2, reward_config.get("lambda2"))
        self.reward_scale = float(
            reward_config.get("reward_scale", 100.0)
            if reward_scale is None
            else reward_scale
        )

        self.features, self.feature_dates, features_from_file = self._load_matrix(
            features,
            default_path="data/processed/features_z_2012-2023.parquet",
        )
        self.regime_probabilities, regime_dates, _ = self._load_matrix(
            regime_probabilities,
            default_path="data/processed/regime_probs_2012-2023.parquet",
        )
        if len(self.features) != len(self.regime_probabilities):
            raise ValueError("features and regime probabilities must have matching rows")
        if isinstance(self.feature_dates, pd.DatetimeIndex) and isinstance(
            regime_dates, pd.DatetimeIndex
        ) and not self.feature_dates.equals(regime_dates):
            raise ValueError("features and regime probabilities must have matching dates")

        if self.features.shape[1] < N_STOCK_FEATURES:
            raise ValueError("features must contain at least 30 columns")
        if self.regime_probabilities.shape[1] < N_REGIME_PROBABILITIES:
            raise ValueError("regime probabilities must contain at least 3 columns")

        if asset_returns is None and features_from_file:
            self.asset_returns = self._load_asset_returns()
        elif asset_returns is None:
            self.asset_returns = np.zeros((len(self.features), 5), dtype=np.float64)
        else:
            self.asset_returns = np.asarray(asset_returns, dtype=np.float64)
        if (
            self.asset_returns.shape != (len(self.features), 5)
            or not np.isfinite(self.asset_returns).all()
            or (self.asset_returns < -1).any()
        ):
            raise ValueError("asset_returns must have one finite simple return per asset and date")

        self._windows = config.get("windows", {})

        obs_dim = self._observation_dim()
        self.action_space = spaces.Discrete(N_ACTIONS)
        observation_low = [-5.0] * N_STOCK_FEATURES
        observation_high = [5.0] * N_STOCK_FEATURES
        if self.include_weights_in_obs:
            observation_low.extend([0.0] * N_BUCKETS)
            observation_high.extend([1.0] * N_BUCKETS)
        if self.regime_enabled:
            observation_low.extend([0.0] * N_REGIME_PROBABILITIES)
            observation_high.extend([1.0] * N_REGIME_PROBABILITIES)
            if self.include_novelty:
                observation_low.append(0.0)
                observation_high.append(2.0)
        self.observation_space = spaces.Box(
            low=np.asarray(observation_low, dtype=np.float32),
            high=np.asarray(observation_high, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.current_index = 0
        self.episode_start_index = 0
        self.episode_length = max(1, self.length)
        self.episode_end_index = min(len(self.features) - 1, self.episode_length)
        self.portfolio_value = self.initial_value
        self.holdings = self.initial_value * self.initial_weights.copy()
        self.weights = self.holdings / self.holdings.sum()
        self.current_date = None
        self.last_action = 0
        self._peak_value = self.initial_value
        self._downside_window: list[float] = []
        self._episode_done = False
        if seed is not None:
            super().reset(seed=seed)
            self.action_space.seed(seed)

    @staticmethod
    def _load_matrix(
        data: ArrayLike | str | Path | None,
        *,
        default_path: str,
    ) -> tuple[NDArray[np.float64], pd.Index, bool]:
        if data is None:
            path = PROJECT_ROOT / default_path
            frame = pd.read_parquet(path)
            return frame.to_numpy(dtype=np.float64), frame.index, True
        if isinstance(data, (str, Path)):
            frame = pd.read_parquet(Path(data))
            return frame.to_numpy(dtype=np.float64), frame.index, True
        if isinstance(data, pd.DataFrame):
            return data.to_numpy(dtype=np.float64), data.index, False
        array = np.asarray(data, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("features and regime probabilities must be two-dimensional")
        return array, pd.RangeIndex(len(array)), False

    def _load_asset_returns(self) -> NDArray[np.float64]:
        prices = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "prices_clean.parquet")
        close = prices.xs("Close", axis=1, level=1).reindex(columns=["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"])
        returns = close.pct_change(fill_method=None).reindex(self.feature_dates)
        if returns.isna().any().any():
            raise ValueError("clean prices do not cover every feature date and its prior close")
        return returns.to_numpy(dtype=np.float64)

    @staticmethod
    def _load_config() -> dict[str, Any]:
        with (PROJECT_ROOT / "config" / "config.yaml").open(encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)

    @staticmethod
    def _coalesce_reward_value(explicit: float | None, configured: Any) -> float:
        value = explicit if explicit is not None else configured
        return 0.0 if value is None else float(value)

    def _resolve_window(self, window: str) -> tuple[int, int]:
        if window not in self._windows:
            raise ValueError(f"unknown environment window: {window}")
        bounds = self._windows[window]
        if len(bounds) != 2 or not isinstance(bounds[0], str):
            raise ValueError(f"window {window!r} is not a single date range")
        if not isinstance(self.feature_dates, pd.DatetimeIndex):
            raise ValueError("date windows require date-indexed feature data")
        start_date, end_date = (pd.Timestamp(value) for value in bounds)
        selected = np.flatnonzero(
            (self.feature_dates >= start_date) & (self.feature_dates <= end_date)
        )
        if selected.size < 2:
            raise ValueError(f"window {window!r} must contain at least two feature dates")
        return int(selected[0]), int(selected[-1] - selected[0])

    def _observation_dim(self) -> int:
        dim = N_STOCK_FEATURES
        if self.include_weights_in_obs:
            dim += N_BUCKETS
        if self.regime_enabled:
            dim += N_REGIME_PROBABILITIES
            if self.include_novelty:
                dim += 1
        return dim

    def _current_feature_vector(self, idx: int) -> NDArray[np.float64]:
        row = self.features[idx]
        return np.asarray(row[:N_STOCK_FEATURES], dtype=np.float64)

    def _current_regime_vector(self, idx: int) -> NDArray[np.float64]:
        row = self.regime_probabilities[idx]
        return np.asarray(row[:N_REGIME_PROBABILITIES], dtype=np.float64)

    def _build_observation(self, idx: int) -> NDArray[np.float32]:
        feature_row = self._current_feature_vector(idx)
        obs = build_observation(
            feature_row,
            self.weights,
            self._current_regime_vector(idx) if self.regime_enabled else None,
            include_weights_in_obs=self.include_weights_in_obs,
            regime_enabled=self.regime_enabled,
            include_novelty=self.include_novelty,
            novelty=0.0 if self.include_novelty else None,
        )
        return obs.astype(np.float32, copy=False)

    def _decode_action(self, action: int) -> dict[str, int | None]:
        return decode_action(action)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        options = {} if options is None else dict(options)
        if "window" in options:
            if "start_idx" in options or "length" in options:
                raise ValueError("provide either window or start_idx/length, not both")
            start_idx, length = self._resolve_window(str(options["window"]))
        else:
            start_idx = int(options.get("start_idx", self.start_idx))
            length = int(options.get("length", self.length))
        if start_idx < 0 or length <= 0 or start_idx + length >= len(self.features):
            raise ValueError("episode start and length must leave a valid next-day transition")

        self.episode_start_index = start_idx
        self.current_index = start_idx
        self.episode_length = length
        self.episode_end_index = start_idx + length
        self.current_step = 0
        self.portfolio_value = self.initial_value
        self.holdings = self.initial_value * self.initial_weights.copy()
        self.weights = self.holdings / self.holdings.sum()
        self._peak_value = self.initial_value
        self._downside_window = []
        self.last_action = 0
        self._episode_done = False

        observation = self._build_observation(self.current_index)
        self.current_date = self.feature_dates[self.current_index]
        info = {
            "date": self.current_date,
            "V": float(self.portfolio_value),
            "cost": 0.0,
            "turnover": 0.0,
            "weights": self.weights.copy(),
            "regime_probs": self._current_regime_vector(self.current_index).copy(),
            "r_gross": 0.0,
            "r_net": 0.0,
            "drawdown": 0.0,
            "sigma_down": 0.0,
            "action_id": 0,
        }
        return observation.astype(np.float32, copy=False), info

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("episode is complete; call reset() before step()")
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer action_id")
        action_id = int(action)
        if action_id < 0 or action_id >= N_ACTIONS:
            raise ValueError(f"action_id must be in [0, {N_ACTIONS - 1}]")

        decoded = self._decode_action(action_id)
        starting_value = self.portfolio_value
        pre_weights = self.weights.copy()
        target_weights = pre_weights.copy()
        if decoded["from"] is not None and decoded["to"] is not None:
            source = int(decoded["from"])
            destination = int(decoded["to"])
            if float(pre_weights[source]) < 1e-9:
                target_weights = pre_weights.copy()
            else:
                trade_fraction = min(self.rebalance_step, float(pre_weights[source]))
                target_weights[source] -= trade_fraction
                target_weights[destination] += trade_fraction
                target_weights = np.clip(target_weights, 0.0, None)
                target_weights /= target_weights.sum()

        trade_result = apply_trades(self.holdings, target_weights, self.cost_model)
        self.holdings = trade_result.holdings.copy()
        next_index = self.current_index + 1
        next_returns = self.asset_returns[next_index]
        self.holdings = mark_to_market(self.holdings, next_returns)
        self.portfolio_value = float(self.holdings.sum())
        self.weights = self.holdings / self.portfolio_value
        truncated = next_index >= self.episode_end_index
        terminated = False

        self._peak_value = max(self._peak_value, self.portfolio_value)
        drawdown = max(0.0, (self._peak_value - self.portfolio_value) / self._peak_value)
        r_gross = float(np.dot(target_weights[:5], next_returns))
        r_net = float((self.portfolio_value - starting_value) / starting_value)
        self._downside_window.append(r_net)
        sigma_down = (
            float(np.mean(np.minimum(self._downside_window[-20:], 0.0) ** 2) ** 0.5)
            if len(self._downside_window) >= 2
            else 0.0
        )
        reward_result = calculate_reward(
            r_gross=r_gross,
            cost=trade_result.cost,
            portfolio_value=starting_value,
            sigma_down=sigma_down,
            drawdown=drawdown,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            reward_scale=self.reward_scale,
        )

        self.current_index = next_index
        self.current_step += 1
        self.last_action = action_id
        self.current_date = self.feature_dates[self.current_index]
        self._episode_done = truncated
        next_observation = self._build_observation(self.current_index)

        info = {
            "date": self.current_date,
            "V": float(self.portfolio_value),
            "cost": float(trade_result.cost),
            "turnover": float(trade_result.turnover),
            "weights": self.weights.copy(),
            "regime_probs": self._current_regime_vector(min(next_index, len(self.regime_probabilities) - 1)).copy(),
            "r_gross": r_gross,
            "r_net": r_net,
            "drawdown": float(drawdown),
            "sigma_down": sigma_down,
            "reward_components": reward_result.as_dict(),
            "action_id": int(action_id),
        }
        return (
            next_observation.astype(np.float32, copy=False),
            reward_result.reward,
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ACTION_TABLE_PATH)
    args = parser.parse_args()
    save_action_table(args.output)
    print(f"Saved {N_ACTIONS}-action table to {args.output}")


if __name__ == "__main__":
    main()
