import json

import numpy as np
import pandas as pd
import pytest

from src.features.scaler import FeatureScaler, fit_configured_scalers


def _features():
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    columns = pd.MultiIndex.from_tuples(
        [("AAA", "x"), ("AAA", "constant"), ("BBB", "y")],
        names=["asset", "feature"],
    )
    values = np.array(
        [
            [1.0, 7.0, 10.0],
            [2.0, 7.0, 20.0],
            [3.0, 7.0, 30.0],
            [4.0, 7.0, 40.0],
            [5.0, 7.0, 50.0],
            [6.0, 7.0, 60.0],
            [1000.0, 7.0, 10000.0],
            [-1000.0, 7.0, -10000.0],
        ]
    )
    return pd.DataFrame(values, index=dates, columns=columns)


def test_t03_scaler_uses_only_fit_window_and_persists(tmp_path):
    features = _features()
    fit_start = "2020-01-01"
    fit_end = "2020-01-06"
    scaler = FeatureScaler().fit(features, fit_start, fit_end)
    artifact = tmp_path / "scaler.json"
    scaler.save(artifact)
    loaded = FeatureScaler.load(artifact)
    transformed = loaded.transform(features)

    fit_slice = features.loc[fit_start:fit_end]
    np.testing.assert_allclose(scaler.means.to_numpy(), fit_slice.mean().to_numpy())
    np.testing.assert_allclose(
        scaler.standard_deviations.to_numpy(),
        fit_slice.std(ddof=0).to_numpy(),
    )
    assert loaded.fit_start == fit_start
    assert loaded.fit_end == fit_end
    assert loaded.fit_rows == len(fit_slice)
    assert transformed.loc[fit_start:fit_end, ("AAA", "x")].mean() == pytest.approx(0)
    assert transformed.loc[fit_start:fit_end, ("AAA", "x")].std(ddof=0) == pytest.approx(1)
    assert transformed.abs().to_numpy().max() <= 5.0
    assert transformed[("AAA", "constant")].eq(0).all()


def test_transform_does_not_change_fitted_parameters_for_out_of_window_data():
    features = _features()
    scaler = FeatureScaler().fit(features, "2020-01-01", "2020-01-06")
    original_means = scaler.means.copy()
    original_stds = scaler.standard_deviations.copy()
    evaluation_data = features.loc["2020-01-07":].copy()
    evaluation_data.loc[:, ("AAA", "x")] = [1e9, -1e9]
    scaler.transform(evaluation_data)

    pd.testing.assert_series_equal(scaler.means, original_means)
    pd.testing.assert_series_equal(scaler.standard_deviations, original_stds)


def test_fit_rejects_empty_or_invalid_window():
    features = _features()
    with pytest.raises(ValueError, match="No feature rows"):
        FeatureScaler().fit(features, "2019-01-01", "2019-12-31")
    with pytest.raises(ValueError, match="must not be after"):
        FeatureScaler().fit(features, "2020-01-06", "2020-01-01")


def test_configured_final_and_tuning_outputs_are_fit_window_locked(tmp_path):
    features_path = tmp_path / "features_raw.parquet"
    config_path = tmp_path / "config.yaml"
    artifacts_dir = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"
    dates = pd.date_range("2012-01-01", "2026-04-30", freq="D")
    columns = pd.MultiIndex.from_product(
        [["NVDA", "AAPL"], ["ret_1d", "vol20"]],
        names=["asset", "feature"],
    )
    t = np.arange(len(dates), dtype=float)
    values = np.column_stack(
        [t, np.sin(t / 20), t * 2, np.cos(t / 15)]
    )
    features = pd.DataFrame(values, index=dates, columns=columns)
    features.loc[features.index >= "2024-01-01"] *= 1e6
    features.to_parquet(features_path)
    config_path.write_text(
        "fit_windows:\n"
        '  final: ["2012-01-01", "2023-12-31"]\n'
        '  tuning: ["2012-01-01", "2021-12-31"]\n',
        encoding="utf-8",
    )

    outputs = fit_configured_scalers(
        features_path, config_path, artifacts_dir, processed_dir
    )

    for label, fit_end in [
        ("final", "2023-12-31"),
        ("tuning", "2021-12-31"),
    ]:
        fit_end_year = fit_end[:4]
        name = f"scaler_2012-{fit_end_year}.json"
        artifact = json.loads((artifacts_dir / name).read_text(encoding="utf-8"))
        raw_fit = features.loc["2012-01-01":fit_end]
        np.testing.assert_allclose(artifact["means"], raw_fit.mean().to_numpy())
        np.testing.assert_allclose(
            artifact["standard_deviations"], raw_fit.std(ddof=0).to_numpy()
        )
        transformed = pd.read_parquet(
            processed_dir / f"features_z_2012-{fit_end_year}.parquet"
        )
        pd.testing.assert_frame_equal(transformed, outputs[label])
        assert transformed.loc[fit_end:].shape[0] > 0
