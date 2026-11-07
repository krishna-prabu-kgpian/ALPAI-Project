from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.agents.curriculum import (
    CurriculumSampler,
    equal_weight_window_drawdowns,
    load_config,
    validate_split_integrity,
)


@pytest.fixture

def training_dates():
    return pd.bdate_range("2012-01-02", "2026-04-30")


def test_t19_stage_boundaries_and_window_mix(training_dates):
    sampler = CurriculumSampler(
        training_dates,
        episode_length=126,
        total_steps=1_000,
        seed=19,
    )
    assert [sampler.stage_at(step) for step in (0, 399, 400, 699, 700)] == [1, 1, 2, 2, 3]

    stage_two = [sampler.sample(400).window for _ in range(5_000)]
    assert set(stage_two) == {"calm", "covid"}
    assert np.mean(np.asarray(stage_two) == "calm") == pytest.approx(0.4, abs=0.03)

    stage_three = [sampler.sample(700).window for _ in range(5_000)]
    observed = {name: np.mean(np.asarray(stage_three) == name) for name in ("calm", "covid", "rates")}
    assert observed == pytest.approx({"calm": 0.3, "covid": 0.3, "rates": 0.4}, abs=0.03)


def test_t19_samples_full_episodes_inside_contiguous_training_segments(training_dates):
    sampler = CurriculumSampler(training_dates, episode_length=126, seed=10)
    for _ in range(300):
        sample = sampler.sample(global_step=0)
        episode_dates = training_dates[sample.start_idx : sample.start_idx + sample.length + 1]
        if sample.window == "calm":
            assert (episode_dates.max() <= pd.Timestamp("2019-12-31") or episode_dates.min() >= pd.Timestamp("2021-01-01"))
        elif sample.window == "covid":
            assert episode_dates.min() >= pd.Timestamp("2020-01-01")
            assert episode_dates.max() <= pd.Timestamp("2020-12-31")
        else:
            assert episode_dates.min() >= pd.Timestamp("2022-01-01")
            assert episode_dates.max() <= pd.Timestamp("2023-12-31")
        assert sample.options == {"start_idx": sample.start_idx, "length": 126}
        assert episode_dates.max() < pd.Timestamp("2024-01-01")


def test_tuning_pool_excludes_2022_and_uses_replacement_stage_mix(training_dates):
    sampler = CurriculumSampler(
        training_dates,
        episode_length=126,
        total_steps=1_000,
        tuning=True,
        seed=12,
    )
    samples = [sampler.sample(global_step=700) for _ in range(2_000)]
    assert {sample.window for sample in samples} == {"calm", "covid"}
    for sample in samples:
        sampled_dates = training_dates[sample.start_idx : sample.start_idx + sample.length + 1]
        assert sampled_dates.max() < pd.Timestamp("2022-01-01")

    observed_calm = np.mean([sample.window == "calm" for sample in samples])
    assert observed_calm == pytest.approx(0.3, abs=0.04)


def test_control_sampler_is_uniform_over_valid_start_days(training_dates):
    sampler = CurriculumSampler(
        training_dates,
        episode_length=126,
        total_steps=1_000,
        curriculum_enabled=False,
        seed=8,
    )
    samples = [sampler.sample(step) for step in range(2_000)]
    assert all(sample.stage == 1 for sample in samples)
    assert sampler.epsilon_at(0) == pytest.approx(1.0)
    assert sampler.epsilon_at(300) == pytest.approx(0.05)
    assert sampler.epsilon_at(900) == pytest.approx(0.05)


def test_t20_split_integrity_and_leak_detection():
    config = load_config()
    validate_split_integrity(config)

    leaked_config = deepcopy(config)
    leaked_config["windows"]["rates"] = [["2025-01-01", "2025-12-31"]]
    with pytest.raises(ValueError, match="precede every test date|excluded window overlaps"):
        validate_split_integrity(leaked_config)

    overlapping_tuning = deepcopy(config)
    overlapping_tuning["windows"]["covid"] = ["2022-01-01", "2022-12-31"]
    with pytest.raises(ValueError, match="tuning pool|val_2022"):
        validate_split_integrity(overlapping_tuning)


def test_t20_excluded_window_never_appears_in_sampled_training_episodes(training_dates):
    config = load_config()
    validate_split_integrity(config)
    sampler = CurriculumSampler(training_dates, episode_length=126, seed=20)
    excluded_start = pd.Timestamp(config["windows"]["excluded"][0])
    excluded_end = pd.Timestamp(config["windows"]["excluded"][1])
    test_start = min(
        pd.Timestamp(config["windows"][name][0])
        for name in ("test_normal", "test_war")
    )

    for step in range(500):
        sample = sampler.sample(step)
        episode_dates = training_dates[sample.start_idx : sample.start_idx + sample.length + 1]
        assert not ((episode_dates >= excluded_start) & (episode_dates <= excluded_end)).any()
        assert episode_dates.max() < test_start


def test_equal_weight_window_drawdowns_resets_equity_for_each_segment():
    dates = pd.bdate_range("2019-12-20", "2021-01-15")
    close = pd.DataFrame(100.0, index=dates, columns=["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"])
    close.loc[pd.Timestamp("2019-12-30"), :] = 80.0
    close.loc[pd.Timestamp("2021-01-05"), :] = 90.0
    windows = {
        "calm": [["2019-12-20", "2019-12-31"], ["2021-01-01", "2021-01-15"]],
        "covid": ["2020-01-01", "2020-12-31"],
        "rates": ["2022-01-01", "2023-12-31"],
    }
    close = pd.concat(
        [
            close.loc[dates <= pd.Timestamp("2020-12-31")],
            pd.DataFrame(
                100.0,
                index=pd.bdate_range("2022-01-03", "2023-12-29"),
                columns=close.columns,
            ),
        ]
    ).sort_index()

    drawdowns = equal_weight_window_drawdowns(close, windows)

    assert drawdowns["calm"] > 0.19
    assert drawdowns["covid"] == pytest.approx(0.0)
    assert drawdowns["rates"] == pytest.approx(0.0)
