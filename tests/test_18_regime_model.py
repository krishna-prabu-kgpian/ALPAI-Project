import joblib
import numpy as np
import pandas as pd
import pytest

from src.regime.build_regime import build_regime_models
from src.regime.regime_model import (
    REGIME_PROBABILITY_COLUMNS,
    RegimeModel,
)


def _synthetic_market_features(rows=700):
    dates = pd.bdate_range("2012-01-03", periods=rows)
    index = np.arange(rows, dtype=float)
    phase = np.sin(index / 16)
    features = pd.DataFrame(
        {
            "m_vol20": 0.18 + 0.13 * (phase > 0) + 0.025 * np.sin(index / 3),
            "m_corr20": 0.12 + 0.55 * (phase > 0) + 0.04 * np.cos(index / 4),
            "m_dd": 0.04 + 0.16 * (phase > 0) + 0.025 * np.sin(index / 5),
        },
        index=dates,
    )
    features.columns.name = "feature"
    return features


def test_18_regime_model():
    features = _synthetic_market_features()
    model = RegimeModel.fit(
        features,
        "2012-01-03",
        features.index[499].date().isoformat(),
    )
    full_probabilities = model.predict_proba(features)
    full_posterior = model.posterior(features)
    assert model.gmm.n_components == 3
    assert model.gmm.covariance_type == "diag"
    assert model.gmm.n_init == 5
    assert model.gmm.random_state == 0
    assert model.ordered_mean_m_vol20[0] <= model.ordered_mean_m_vol20[1]
    assert model.ordered_mean_m_vol20[1] <= model.ordered_mean_m_vol20[2]
    assert tuple(full_probabilities.columns) == REGIME_PROBABILITY_COLUMNS
    np.testing.assert_allclose(full_probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert (full_probabilities.to_numpy() >= 0).all()

    cut_points = np.random.default_rng(1801).choice(
        np.arange(60, len(features)), size=50, replace=False
    )
    for cut in cut_points:
        date = features.index[cut]
        truncated_features = features.iloc[: cut + 1]
        truncated_posterior = model.posterior(truncated_features)
        np.testing.assert_allclose(
            truncated_posterior.loc[date].to_numpy(),
            full_posterior.loc[date].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
        truncated = model.predict_proba(features.iloc[: cut + 1])
        np.testing.assert_allclose(
            truncated.loc[date].to_numpy(),
            full_probabilities.loc[date].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )


def test_regime_model_fit_statistics_ignore_rows_after_fit_end():
    features = _synthetic_market_features()
    fit_end = features.index[499].date().isoformat()
    changed = features.copy()
    changed.loc[changed.index > fit_end] *= 10000

    first = RegimeModel.fit(features, "2012-01-03", fit_end)
    second = RegimeModel.fit(changed, "2012-01-03", fit_end)

    np.testing.assert_allclose(first.scaler.means, second.scaler.means)
    np.testing.assert_allclose(
        first.scaler.standard_deviations, second.scaler.standard_deviations
    )
    np.testing.assert_allclose(first.gmm.means_, second.gmm.means_)


def test_build_saves_both_models_probabilities_and_pretest_report(tmp_path):
    features = _synthetic_market_features(rows=700)
    features_path = tmp_path / "market_features.parquet"
    config_path = tmp_path / "config.yaml"
    artifacts_dir = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"
    report_path = tmp_path / "regime_report.md"
    features.to_parquet(features_path)
    config_path.write_text(
        "fit_windows:\n"
        '  final: ["2012-01-03", "2013-12-31"]\n'
        '  tuning: ["2012-01-03", "2013-06-30"]\n'
        "regime:\n"
        "  k: 3\n"
        "  covariance: diag\n"
        "  n_init: 5\n"
        "  random_state: 0\n"
        "  ema_span: 5\n"
        "windows:\n"
        '  calm: [["2012-01-03", "2012-12-31"]]\n'
        '  covid: [["2013-01-01", "2013-03-31"]]\n'
        '  rates: [["2013-04-01", "2013-12-31"]]\n'
        '  val_2022: ["2013-07-01", "2013-07-31"]\n',
        encoding="utf-8",
    )

    outputs = build_regime_models(
        features_path, config_path, artifacts_dir, processed_dir, report_path
    )

    for label, window_label in (
        ("final", "2012-2023"),
        ("tuning", "2012-2021"),
    ):
        model = joblib.load(artifacts_dir / f"regime_{window_label}.joblib")
        saved = pd.read_parquet(
            processed_dir / f"regime_probs_{window_label}.parquet"
        )
        pd.testing.assert_frame_equal(saved, outputs[label])
        np.testing.assert_allclose(saved.sum(axis=1), 1.0, atol=1e-12)
        assert model.fit_end < "2014-12-31"
    report = report_path.read_text(encoding="utf-8")
    assert "val_2022" in report
    assert report.count("| val_2022 |") == 1
    assert "2026" not in report


def test_market_features_reject_incorrect_column_order():
    features = _synthetic_market_features()
    features = features[["m_dd", "m_vol20", "m_corr20"]]
    with pytest.raises(ValueError, match="must be ordered"):
        RegimeModel.fit(features, "2012-01-03", "2013-12-31")
