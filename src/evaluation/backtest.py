"""Deterministic greedy evaluation passes, degeneracy guard and the test-window freeze check."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.agents.curriculum import load_config
from src.benchmarks.simulator import FIT_LABELS, record_row
from src.evaluation.diagnostics import summarize_episode
from src.evaluation.metrics import drawdown_series, rolling_sharpe, rolling_sortino
from src.env.portfolio_env import PortfolioEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
FROZEN_PATH = PROJECT_ROOT / "config" / "FROZEN"
TEST_WINDOWS = ("test_normal", "test_war")

# RW-04 thresholds; entropy below this counts as "approximately zero".
MAX_MEAN_CASH = 0.80
MAX_HOLD_FRACTION = 0.95
MIN_ACTION_ENTROPY = 1e-3
MAX_MEAN_TURNOVER = 0.5


class Policy(Protocol):
    def greedy_action(self, observation: NDArray[np.float32]) -> int: ...


def config_hash(path: Path = CONFIG_PATH) -> str:
    """SHA-256 of the config file bytes; stored in checkpoints and FROZEN."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_freeze(window: str, frozen_path: Path = FROZEN_PATH, config_path: Path = CONFIG_PATH) -> None:
    """S-03/B-01: refuse to score a test window without a FROZEN tag matching the config."""
    if window not in TEST_WINDOWS:
        return
    if not Path(frozen_path).exists():
        raise RuntimeError(f"{window} is blind: config/FROZEN does not exist")
    frozen = Path(frozen_path).read_text(encoding="utf-8").split()
    if config_hash(config_path) not in frozen:
        raise RuntimeError(f"{window} is blind: config/FROZEN does not match config.yaml")


def make_env(
    fit: str = "final",
    *,
    regime_enabled: bool = True,
    lambda1: float | None = None,
    lambda2: float | None = None,
    config: dict[str, Any] | None = None,
) -> PortfolioEnv:
    """Build the env on the fit window's z-scored features and regime probabilities."""
    config = load_config() if config is None else config
    label = FIT_LABELS[fit]
    processed = PROJECT_ROOT / "data" / "processed"
    env_config = config["env"]
    return PortfolioEnv(
        processed / f"features_z_{label}.parquet",
        processed / f"regime_probs_{label}.parquet",
        length=int(env_config["episode_len"]),
        include_weights_in_obs=bool(env_config["include_weights_in_obs"]),
        regime_enabled=regime_enabled,
        cost=float(env_config["cost"]),
        rebalance_step=float(env_config["step"]),
        lambda1=lambda1,
        lambda2=lambda2,
    )


def greedy_pass(policy: Policy, env: PortfolioEnv, window: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """One epsilon = 0 pass over ``window``; returns C-01 runner rows and raw infos."""
    require_freeze(window)
    observation, info = env.reset(options={"window": window})
    infos: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    previous_value = info["V"]
    truncated = False
    while not truncated:
        action = policy.greedy_action(observation)
        observation, _, _, truncated, info = env.step(action)
        infos.append(info)
        rows.append(
            record_row(
                info["date"],
                info["V"],
                info["V"] / previous_value - 1.0,
                info["cost"],
                info["turnover"],
                info["weights"],
                info["regime_probs"],
            )
        )
        previous_value = info["V"]
    return pd.DataFrame(rows), infos


def degeneracy_guard(infos: list[dict[str, Any]]) -> dict[str, float | bool]:
    """RW-04: the four logged numbers plus the pass flag."""
    summary = summarize_episode(infos)
    passed = (
        summary["mean_cash"] <= MAX_MEAN_CASH
        and summary["hold_fraction"] <= MAX_HOLD_FRACTION
        and summary["action_entropy"] >= MIN_ACTION_ENTROPY
        and summary["mean_turnover"] <= MAX_MEAN_TURNOVER
    )
    return {**summary, "guard_passed": bool(passed)}


def finalize_frame(frame: pd.DataFrame, regime_label: str) -> pd.DataFrame:
    """Add the four exporter columns of C-01 (regime label, drawdown, rolling 21-day ratios)."""
    result = frame.copy()
    returns = result["daily_return"].to_numpy()
    result["regime"] = regime_label
    result["drawdown"] = drawdown_series(returns)
    result["rolling_sharpe_21"] = rolling_sharpe(returns)
    result["rolling_sortino_21"] = rolling_sortino(returns)
    return result
