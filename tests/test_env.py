import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.env.portfolio_env import (
    build_action_table,
    build_observation,
    decode_action,
    save_action_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_t04_action_table_bijection_and_json_export(tmp_path):
    actions = build_action_table()
    assert len(actions) == 31
    assert actions[0] == {"action_id": 0, "from": None, "to": None}
    assert decode_action(0) == actions[0]
    pairs = [(action["from"], action["to"]) for action in actions[1:]]
    assert len(set(pairs)) == 30
    assert all(source != destination for source, destination in pairs)
    assert pairs == [
        (source, destination)
        for source in range(6)
        for destination in range(6)
        if source != destination
    ]

    output_path = tmp_path / "action_table.json"
    save_action_table(output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == actions


def test_observation_matches_feature_and_regime_parquets():
    features = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "features_z_2012-2023.parquet"
    )
    regimes = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "regime_probs_2012-2023.parquet"
    )
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
    observation = build_observation(
        features.iloc[0].to_numpy(),
        weights,
        regimes.iloc[0].to_numpy(),
    )
    expected = np.concatenate(
        [
            features.iloc[0].to_numpy(dtype=np.float32),
            weights.astype(np.float32),
            regimes.iloc[0].to_numpy(dtype=np.float32),
        ]
    )
    assert observation.dtype == np.float32
    np.testing.assert_array_equal(observation, expected)


def test_observation_configuration_toggles():
    features = np.zeros(30)
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0])
    probabilities = np.array([0.2, 0.3, 0.5])

    assert build_observation(features, weights, probabilities).shape == (39,)
    assert build_observation(
        features, weights, probabilities, include_weights_in_obs=False
    ).shape == (33,)
    assert build_observation(
        features, weights, None, regime_enabled=False
    ).shape == (36,)
    assert build_observation(
        features,
        weights,
        probabilities,
        include_novelty=True,
        novelty=0.75,
    ).shape == (40,)
