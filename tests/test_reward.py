import numpy as np
import pytest

from src.env.reward import calculate_reward
from src.evaluation.diagnostics import summarize_episode


def test_t09_reward_matches_specification_with_single_cost_deduction():
    result = calculate_reward(
        r_gross=0.012,
        cost=0.0005,
        portfolio_value=1.0,
        sigma_down=0.004,
        drawdown=0.03,
        lambda1=0.5,
        lambda2=0.2,
        reward_scale=100.0,
    )

    expected = 100.0 * (0.012 - 0.0005 / 1.0 - 0.5 * 0.004 - 0.2 * 0.03)
    assert result.reward == pytest.approx(expected, abs=1e-6)
    assert result.cost_fraction == pytest.approx(0.0005)
    assert result.lambda1_penalty == pytest.approx(0.002)
    assert result.lambda2_penalty == pytest.approx(0.006)
    assert result.as_dict()["r_gross"] == pytest.approx(0.012)


def test_reward_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="positive"):
        calculate_reward(
            r_gross=0.01,
            cost=0.0,
            portfolio_value=0.0,
            sigma_down=0.0,
            drawdown=0.0,
            lambda1=0.0,
            lambda2=0.0,
        )


def test_episode_diagnostics_summarize_recorded_info():
    summary = summarize_episode(
        [
            {
                "weights": np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1]),
                "turnover": 0.2,
                "action_id": 0,
            },
            {
                "weights": np.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2]),
                "turnover": 0.4,
                "action_id": 1,
            },
        ]
    )

    assert summary["mean_cash"] == pytest.approx(0.15)
    assert summary["mean_turnover"] == pytest.approx(0.3)
    assert summary["hold_fraction"] == pytest.approx(0.5)
    assert summary["action_entropy"] == pytest.approx(np.log(2.0))
