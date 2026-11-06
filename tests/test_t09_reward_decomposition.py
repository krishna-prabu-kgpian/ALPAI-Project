import numpy as np

from src.env.portfolio_env import PortfolioEnv


def test_t09_environment_reports_reward_decomposition():
    features = np.zeros((3, 30))
    regimes = np.tile([0.2, 0.3, 0.5], (3, 1))
    returns = np.zeros((3, 5))
    returns[1] = [-0.05, 0.0, 0.0, 0.0, 0.0]
    env = PortfolioEnv(
        features,
        regimes,
        asset_returns=returns,
        length=2,
        cost=0.0,
        lambda1=0.5,
        lambda2=0.2,
        reward_scale=100.0,
    )
    env.reset(seed=9)

    _, reward, _, _, info = env.step(0)
    components = info["reward_components"]
    expected = 100.0 * (
        components["r_gross"]
        - components["cost_fraction"]
        - components["lambda1_penalty"]
        - components["lambda2_penalty"]
    )

    assert abs(reward - expected) < 1e-6
    assert components["lambda1_penalty"] == 0.0
    assert components["lambda2_penalty"] > 0.0
