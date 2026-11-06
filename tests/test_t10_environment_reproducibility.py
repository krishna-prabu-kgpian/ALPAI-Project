import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from src.env.portfolio_env import PortfolioEnv


def test_t10_environment_reproducibility():
    env = PortfolioEnv()
    check_env(env, skip_render_check=True)

    env1 = PortfolioEnv()
    env2 = PortfolioEnv()
    obs1, info1 = env1.reset(seed=7)
    obs2, info2 = env2.reset(seed=7)
    np.testing.assert_array_equal(obs1, obs2)
    assert info1.keys() == info2.keys()
    for key in info1:
        if isinstance(info1[key], np.ndarray):
            np.testing.assert_array_equal(info1[key], info2[key])
        else:
            assert info1[key] == info2[key]

    action = 0
    next_obs, reward, terminated, truncated, info = env1.step(action)
    assert next_obs.shape == env1.observation_space.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert isinstance(info, dict)
    assert set(
        [
            "date",
            "V",
            "cost",
            "turnover",
            "weights",
            "regime_probs",
            "r_gross",
            "r_net",
            "drawdown",
            "sigma_down",
            "action_id",
        ]
    ).issubset(info)


def test_step_applies_next_day_returns_and_truncates_at_episode_end():
    features = np.zeros((4, 30), dtype=np.float64)
    regimes = np.tile([0.2, 0.3, 0.5], (4, 1))
    asset_returns = np.zeros((4, 5), dtype=np.float64)
    asset_returns[1, 0] = 0.10
    env = PortfolioEnv(
        features,
        regimes,
        asset_returns=asset_returns,
        length=1,
        cost=0.0,
    )

    observation, _ = env.reset(seed=2, options={"start_idx": 0, "length": 1})
    next_observation, reward, terminated, truncated, info = env.step(0)

    assert observation.shape == next_observation.shape == (39,)
    assert terminated is False
    assert truncated is True
    np.testing.assert_allclose(info["V"], 1.02, atol=1e-12)
    np.testing.assert_allclose(info["r_gross"], 0.02, atol=1e-12)
    np.testing.assert_allclose(info["r_net"], 0.02, atol=1e-12)
    np.testing.assert_allclose(reward, 2.0, atol=1e-12)
    np.testing.assert_allclose(info["weights"].sum(), 1.0, atol=1e-12)


def test_reset_accepts_configured_evaluation_window():
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    features = pd.DataFrame(np.zeros((4, 30)), index=dates)
    regimes = np.tile([0.2, 0.3, 0.5], (4, 1))
    returns = np.zeros((4, 5))
    env = PortfolioEnv(features, regimes, asset_returns=returns)

    observation, info = env.reset(options={"window": "test_normal"})

    assert observation.shape == (39,)
    assert info["date"] == dates[0]
    for step_index in range(len(dates) - 1):
        _, _, terminated, truncated, info = env.step(0)
        assert terminated is False
        assert truncated is (step_index == len(dates) - 2)
    assert info["date"] == dates[-1]


def test_t05_weights_remain_valid_for_10000_random_actions():
    row_count = 10_002
    features = np.zeros((row_count, 30))
    regimes = np.tile([0.2, 0.3, 0.5], (row_count, 1))
    returns = np.zeros((row_count, 5))
    env = PortfolioEnv(features, regimes, asset_returns=returns, length=10_000)
    env.reset(seed=17)

    for step_index in range(10_000):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        weights = info["weights"]
        assert np.all(weights >= 0)
        assert abs(float(weights.sum()) - 1.0) < 1e-9
        assert terminated is False
        assert truncated is (step_index == 9_999)
