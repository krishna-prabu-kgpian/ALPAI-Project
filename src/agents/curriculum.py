"""Curriculum episode sampling and temporal split-integrity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_WINDOWS = ("calm", "covid", "rates")
TUNING_WINDOWS = ("calm", "covid")
SCORED_WINDOWS = ("val_2022", "test_normal", "test_war")


@dataclass(frozen=True)
class EpisodeSample:
    """A sampled episode and its curriculum metadata."""

    start_idx: int
    length: int
    window: str
    stage: int

    @property
    def options(self) -> dict[str, int]:
        """Options accepted by PortfolioEnv.reset()."""
        return {"start_idx": self.start_idx, "length": self.length}


class CurriculumSampler:
    """Sample whole episodes from fixed curriculum windows."""

    def __init__(
        self,
        dates: pd.DatetimeIndex,
        *,
        config: dict[str, Any] | None = None,
        episode_length: int | None = None,
        total_steps: int | None = None,
        tuning: bool = False,
        curriculum_enabled: bool | None = None,
        seed: int | None = None,
    ) -> None:
        if not isinstance(dates, pd.DatetimeIndex) or dates.empty:
            raise ValueError("dates must be a non-empty DatetimeIndex")
        if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("dates must be unique, non-null, and strictly increasing")
        self.config = load_config() if config is None else config
        validate_split_integrity(self.config)
        self.dates = dates
        self.windows = self.config["windows"]
        self.curriculum_config = self.config.get("curriculum", {})
        self.tuning = bool(tuning)
        configured_length = self.config.get("env", {}).get("episode_len", 126)
        self.episode_length = int(configured_length if episode_length is None else episode_length)
        if self.episode_length < 1:
            raise ValueError("episode_length must be positive")
        agents_config = self.config.get("agents", {})
        configured_steps = (
            agents_config.get("tuning_steps", 75_000)
            if self.tuning
            else agents_config.get("train_steps", 150_000)
        )
        self.total_steps = int(configured_steps if total_steps is None else total_steps)
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        configured_enabled = self.curriculum_config.get("enabled", True)
        self.curriculum_enabled = (
            bool(configured_enabled)
            if curriculum_enabled is None
            else bool(curriculum_enabled)
        )
        self.rng = np.random.default_rng(seed)
        self._valid_starts = self._build_valid_starts()
        self.last_sample: EpisodeSample | None = None

    def stage_at(self, global_step: int) -> int:
        """Return the 1-based curriculum stage for a zero-based global step."""
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        if not self.curriculum_enabled:
            return 1
        shares = np.asarray(self.curriculum_config.get("stage_share", [0.4, 0.3, 0.3]), dtype=float)
        if shares.shape != (3,) or not np.isfinite(shares).all() or (shares < 0).any():
            raise ValueError("curriculum stage_share must contain three non-negative values")
        if not np.isclose(shares.sum(), 1.0):
            raise ValueError("curriculum stage_share must sum to 1")
        boundaries = np.cumsum(shares)[:-1] * self.total_steps
        return int(np.searchsorted(boundaries, global_step, side="right") + 1)

    def epsilon_at(self, global_step: int) -> float:
        """Return the CU-05/CU-07 linear exploration rate."""
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        curriculum = self.curriculum_config
        epsilon_end = float(curriculum.get("eps_end", 0.05))
        if not self.curriculum_enabled:
            decay_steps = max(1.0, self.total_steps * 0.30)
            return max(epsilon_end, 1.0 - (1.0 - epsilon_end) * global_step / decay_steps)

        shares = np.asarray(curriculum.get("stage_share", [0.4, 0.3, 0.3]), dtype=float)
        boundaries = np.concatenate(([0.0], np.cumsum(shares) * self.total_steps))
        stage = self.stage_at(global_step)
        stage_index = min(stage - 1, 2)
        stage_start = boundaries[stage_index]
        stage_length = boundaries[stage_index + 1] - stage_start
        start_values = curriculum.get("eps_stage_start", [1.0, 0.2, 0.2])
        epsilon_start = float(start_values[stage_index])
        decay_fraction = float(curriculum.get("eps_decay_frac_of_stage", 0.5))
        decay_steps = max(1.0, stage_length * decay_fraction)
        progress = min(1.0, max(0.0, (global_step - stage_start) / decay_steps))
        return max(epsilon_end, epsilon_start + (epsilon_end - epsilon_start) * progress)

    def sample(self, global_step: int, *, episode_length: int | None = None) -> EpisodeSample:
        """Sample a window and a valid start index for one complete episode."""
        length = self.episode_length if episode_length is None else int(episode_length)
        if length < 1:
            raise ValueError("episode_length must be positive")
        stage = self.stage_at(global_step)
        if not self.curriculum_enabled:
            starts = np.concatenate(
                [self._valid_starts[name] for name in self._pool_names()]
            )
            start_idx = int(self.rng.choice(starts))
            window = self._window_for_start(start_idx)
        else:
            window = self._sample_window(stage)
            starts = self._valid_starts_for_length(window, length)
            if starts.size == 0:
                raise ValueError(
                    f"window {window!r} has no valid starts for episode length {length}"
                )
            start_idx = int(self.rng.choice(starts))
        sample = EpisodeSample(start_idx, length, window, stage)
        self.last_sample = sample
        return sample

    def _pool_names(self) -> tuple[str, ...]:
        return TUNING_WINDOWS if self.tuning else TRAINING_WINDOWS

    def _build_valid_starts(self) -> dict[str, NDArray[np.int64]]:
        result: dict[str, NDArray[np.int64]] = {}
        for name in self._pool_names():
            if name not in self.windows:
                raise ValueError(f"missing training window {name!r} in config")
            segments = _parse_window_segments(self.windows[name])
            valid: list[NDArray[np.int64]] = []
            for start_date, end_date in segments:
                indices = np.flatnonzero(
                    (self.dates >= start_date) & (self.dates <= end_date)
                )
                if indices.size <= self.episode_length:
                    continue
                positions = np.arange(indices[0], indices[-1] - self.episode_length + 1)
                valid.append(positions.astype(np.int64, copy=False))
            result[name] = np.concatenate(valid) if valid else np.empty(0, dtype=np.int64)
            if result[name].size == 0:
                raise ValueError(
                    f"training window {name!r} has no valid episode starts for length "
                    f"{self.episode_length}"
                )
        return result

    def _valid_starts_for_length(self, window: str, length: int) -> NDArray[np.int64]:
        if length == self.episode_length:
            return self._valid_starts[window]
        starts: list[NDArray[np.int64]] = []
        for start_date, end_date in _parse_window_segments(self.windows[window]):
            indices = np.flatnonzero((self.dates >= start_date) & (self.dates <= end_date))
            if indices.size > length:
                starts.append(
                    np.arange(indices[0], indices[-1] - length + 1, dtype=np.int64)
                )
        return np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)

    def _sample_window(self, stage: int) -> str:
        curriculum = self.curriculum_config
        names = list(TRAINING_WINDOWS)
        if self.tuning:
            if stage == 3:
                probabilities = curriculum.get("tuning_stage3_mix", [0.3, 0.7, 0.0])
                names = ["calm", "covid", "rates"]
            else:
                probabilities = curriculum.get("stage_mix", [])[stage - 1]
        else:
            probabilities = curriculum.get("stage_mix", [])[stage - 1]
        probabilities_array = np.asarray(probabilities, dtype=np.float64)
        if probabilities_array.shape != (len(names),) or not np.isfinite(probabilities_array).all():
            raise ValueError(f"invalid window mix for curriculum stage {stage}")
        if (probabilities_array < 0).any() or not np.isclose(probabilities_array.sum(), 1.0):
            raise ValueError(f"window mix for stage {stage} must be non-negative and sum to 1")
        if self.tuning:
            probabilities_array[2] = 0.0
            probabilities_array[:2] /= probabilities_array[:2].sum()
        available = np.asarray([self._valid_starts.get(name, np.empty(0)).size > 0 for name in names])
        probabilities_array *= available
        if probabilities_array.sum() == 0:
            raise ValueError(f"no available windows in curriculum stage {stage}")
        probabilities_array /= probabilities_array.sum()
        return str(self.rng.choice(names, p=probabilities_array))

    def _window_for_start(self, start_idx: int) -> str:
        candidates = [
            name for name, starts in self._valid_starts.items() if np.any(starts == start_idx)
        ]
        if not candidates:
            raise RuntimeError(f"sampled start index {start_idx} is outside training windows")
        return candidates[0]


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the project YAML configuration."""
    config_path = PROJECT_ROOT / "config" / "config.yaml" if path is None else Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"configuration at {config_path} must be a YAML mapping")
    return config


def validate_split_integrity(config: dict[str, Any]) -> None:
    """Reject configurations where training, fitting, or tuning overlap held-out data."""
    windows = config.get("windows", {})
    fit_windows = config.get("fit_windows", {})
    required_windows = (*TRAINING_WINDOWS, *SCORED_WINDOWS, "excluded")
    missing = [name for name in required_windows if name not in windows]
    if missing:
        raise ValueError(f"missing required split windows: {', '.join(missing)}")

    training_segments = [segment for name in TRAINING_WINDOWS for segment in _parse_window_segments(windows[name])]
    fit_final = _parse_window_segments(fit_windows.get("final"))
    fit_tuning = _parse_window_segments(fit_windows.get("tuning"))
    excluded_segments = _parse_window_segments(windows["excluded"])
    test_segments = _parse_window_segments(windows["test_normal"]) + _parse_window_segments(windows["test_war"])
    tuning_segments = [segment for name in TUNING_WINDOWS for segment in _parse_window_segments(windows[name])]

    if not fit_final or not fit_tuning:
        raise ValueError("final and tuning fit windows must be configured")
    earliest_test = min(start for start, _ in test_segments)
    latest_training = max(end for _, end in training_segments)
    latest_fit = max(end for _, end in fit_final)
    if latest_training >= earliest_test or latest_fit >= earliest_test:
        raise ValueError("training and final-fit dates must precede every test date")

    tuning_fit_end = max(end for _, end in fit_tuning)
    if max(end for _, end in tuning_segments) > tuning_fit_end:
        raise ValueError("tuning pool extends beyond the tuning fit window")
    val_segments = _parse_window_segments(windows["val_2022"])
    if _segments_overlap(tuning_segments, val_segments):
        raise ValueError("tuning pool must exclude val_2022")

    protected_segments = training_segments + tuning_segments + fit_final + fit_tuning + test_segments + val_segments
    if _segments_overlap(protected_segments, excluded_segments):
        raise ValueError("excluded window overlaps training, fitting, tuning, or scored windows")


def equal_weight_window_drawdowns(
    prices: pd.DataFrame,
    windows: dict[str, Any],
    *,
    tickers: tuple[str, ...] = ("NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"),
) -> dict[str, float]:
    """Calculate max drawdown of the daily-rebalanced equal-weight basket by segment."""
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.has_duplicates:
        raise ValueError("prices must have a unique DatetimeIndex")
    if isinstance(prices.columns, pd.MultiIndex):
        close = prices.xs("Close", axis=1, level=1).reindex(columns=list(tickers))
    else:
        close = prices.reindex(columns=list(tickers))
    if close.isna().any().any() or (close <= 0).any().any():
        raise ValueError("prices must contain finite positive close values for all tickers")
    basket_returns = close.pct_change(fill_method=None).mean(axis=1)
    result: dict[str, float] = {}
    for name in TRAINING_WINDOWS:
        if name not in windows:
            raise ValueError(f"missing training window {name!r}")
        segment_drawdowns: list[float] = []
        for start_date, end_date in _parse_window_segments(windows[name]):
            returns = basket_returns.loc[(prices.index >= start_date) & (prices.index <= end_date)]
            if len(returns) < 2:
                continue
            equity = np.concatenate(([1.0], np.cumprod(1.0 + returns.iloc[1:].to_numpy())))
            running_peak = np.maximum.accumulate(equity)
            segment_drawdowns.append(float((1.0 - equity / running_peak).max()))
        if not segment_drawdowns:
            raise ValueError(f"no price rows found for training window {name!r}")
        result[name] = max(segment_drawdowns)
    return result


def _parse_window_segments(window_spec: Any) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if isinstance(window_spec, (list, tuple)) and len(window_spec) == 2 and all(
        isinstance(value, str) for value in window_spec
    ):
        candidates = [window_spec]
    elif isinstance(window_spec, (list, tuple)) and all(
        isinstance(item, (list, tuple)) and len(item) == 2 for item in window_spec
    ):
        candidates = list(window_spec)
    else:
        raise ValueError(f"invalid date window specification: {window_spec!r}")
    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in candidates:
        start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
        if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
            raise ValueError(f"invalid date bounds: {start!r}, {end!r}")
        segments.append((start_date, end_date))
    return segments


def _segments_overlap(
    left: list[tuple[pd.Timestamp, pd.Timestamp]],
    right: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> bool:
    return any(
        left_start <= right_end and right_start <= left_end
        for left_start, left_end in left
        for right_start, right_end in right
    )
