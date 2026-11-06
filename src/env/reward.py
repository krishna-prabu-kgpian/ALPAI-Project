"""Risk-sensitive portfolio reward calculation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RewardResult:
    """Scaled reward and the raw components used to calculate it."""

    reward: float
    r_gross: float
    cost_fraction: float
    sigma_down: float
    drawdown: float
    lambda1_penalty: float
    lambda2_penalty: float
    reward_scale: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_reward(
    *,
    r_gross: float,
    cost: float,
    portfolio_value: float,
    sigma_down: float,
    drawdown: float,
    lambda1: float,
    lambda2: float,
    reward_scale: float = 100.0,
) -> RewardResult:
    """Calculate RW-01, deducting transaction cost exactly once."""
    values = np.asarray(
        [r_gross, cost, portfolio_value, sigma_down, drawdown, lambda1, lambda2, reward_scale],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("reward inputs must all be finite")
    if cost < 0 or portfolio_value <= 0 or sigma_down < 0 or drawdown < 0:
        raise ValueError("cost, risk measures, and drawdown must be non-negative; value positive")
    if lambda1 < 0 or lambda2 < 0 or reward_scale <= 0:
        raise ValueError("reward penalties must be non-negative and scale positive")

    cost_fraction = float(cost / portfolio_value)
    lambda1_penalty = float(lambda1 * sigma_down)
    lambda2_penalty = float(lambda2 * drawdown)
    reward = float(
        reward_scale * (r_gross - cost_fraction - lambda1_penalty - lambda2_penalty)
    )
    return RewardResult(
        reward=reward,
        r_gross=float(r_gross),
        cost_fraction=cost_fraction,
        sigma_down=float(sigma_down),
        drawdown=float(drawdown),
        lambda1_penalty=lambda1_penalty,
        lambda2_penalty=lambda2_penalty,
        reward_scale=float(reward_scale),
    )