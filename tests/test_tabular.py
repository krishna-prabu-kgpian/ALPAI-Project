import itertools

import numpy as np

from src.agents.tabular_q import (
    N_STATES,
    TabularQAgent,
    fit_tabular_cuts,
    sma50_columns,
    state_components,
    state_index,
)

CUTS = {
    "fit_window": "synthetic",
    "trend_terciles": [-0.01, 0.02],
    "cash_edges": [0.05, 0.5],
    "sma50_mean": [0.0] * 5,
    "sma50_std": [1.0] * 5,
}
GAP_BY_TREND = {0: -0.05, 1: 0.0, 2: 0.05}
CASH_BY_BUCKET = {0: 0.0, 1: 0.2, 2: 0.6}


def _observation(regime: int, trend: int, leader: int, cash_bucket: int) -> np.ndarray:
    obs = np.zeros(39, dtype=np.float32)
    gaps = np.full(5, GAP_BY_TREND[trend])
    gaps[leader] += 0.004
    gaps -= 0.004 / 5  # keep the mean inside its tercile
    obs[sma50_columns()] = gaps
    cash = CASH_BY_BUCKET[cash_bucket]
    obs[30:36] = np.append(np.full(5, (1.0 - cash) / 5), cash)
    obs[36 + regime] = 1.0
    return obs


def test_all_135_states_are_reachable_and_mapping_is_bijective():
    seen = {}
    for components in itertools.product(range(3), range(3), range(5), range(3)):
        obs = _observation(*components)
        assert state_components(obs, CUTS) == components
        index = state_index(obs, CUTS)
        assert state_index(obs.copy(), CUTS) == index
        seen[index] = components
    assert sorted(seen) == list(range(N_STATES))


def test_cash_bucket_edges():
    obs = _observation(0, 1, 0, 0)
    for cash, bucket in ((0.0499, 0), (0.05, 1), (0.4999, 1), (0.5, 2), (1.0, 2)):
        obs[30:36] = np.append(np.full(5, (1.0 - cash) / 5), cash)
        assert state_components(obs, CUTS)[3] == bucket


def test_cuts_use_fit_window_only_and_inverse_z_score():
    cuts = fit_tabular_cuts("tuning")
    lower, upper = cuts["trend_terciles"]
    assert lower < upper
    assert len(cuts["sma50_mean"]) == len(cuts["sma50_std"]) == 5
    assert cuts["fit_window"] == "2012-2021"


def test_q_update_follows_alpha_and_gamma():
    agent = TabularQAgent(CUTS, alpha=0.1, gamma=0.99, seed=0)
    obs = _observation(0, 0, 0, 0)
    nxt = _observation(2, 2, 4, 2)
    agent.q_table[state_index(nxt, CUTS), 3] = 2.0
    agent.observe(obs, 7, 1.0, nxt, False)
    assert agent.q_table[state_index(obs, CUTS), 7] == 0.1 * (1.0 + 0.99 * 2.0)
