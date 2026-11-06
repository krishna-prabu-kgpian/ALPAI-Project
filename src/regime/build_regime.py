"""Fit and report the final and tuning market-regime models."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.regime.regime_model import (
    MARKET_FEATURES,
    REGIME_NAMES,
    RegimeModel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "market_features_raw.parquet"
)
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "regime_report.md"
FIT_WINDOWS = {
    "final": ("final", "2012-2023"),
    "tuning": ("tuning", "2012-2021"),
}


def build_regime_models(
    features_path: Path = DEFAULT_FEATURES_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, pd.DataFrame]:
    """Fit each configured model on its fit dates and predict continuously."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    try:
        fit_windows = config["fit_windows"]
        regime_config = config["regime"]
        window_config = config["windows"]
        n_components = regime_config["k"]
        covariance_type = regime_config["covariance"]
        n_init = regime_config["n_init"]
        random_state = regime_config["random_state"]
        ema_span = regime_config["ema_span"]
    except (TypeError, KeyError) as error:
        raise ValueError(f"Missing regime configuration in {config_path}") from error
    if covariance_type != "diag":
        raise ValueError("The regime specification requires covariance_type: diag")
    if not regime_config.get("enabled", True):
        raise ValueError(
            "Regime modeling is disabled by config regime.enabled; "
            "do not build regime artifacts for control A1"
        )

    features = pd.read_parquet(features_path)
    _validate_feature_range(features)
    probabilities: dict[str, pd.DataFrame] = {}
    models: dict[str, RegimeModel] = {}

    for label, (config_key, window_label) in FIT_WINDOWS.items():
        if config_key not in fit_windows:
            raise ValueError(f"Missing fit_windows.{config_key} in {config_path}")
        fit_start, fit_end = fit_windows[config_key]
        model = RegimeModel.fit(
            features,
            fit_start,
            fit_end,
            n_components=n_components,
            n_init=n_init,
            random_state=random_state,
            ema_span=ema_span,
        )
        model_path = Path(artifacts_dir) / f"regime_{window_label}.joblib"
        output_path = Path(processed_dir) / f"regime_probs_{window_label}.parquet"
        _atomic_joblib_dump(model, model_path)
        loaded_model: RegimeModel = joblib.load(model_path)
        smoothed = loaded_model.predict_proba(features)
        _atomic_write_parquet(smoothed, output_path)
        probabilities[label] = smoothed
        models[label] = loaded_model

    _write_report(
        features,
        probabilities,
        models,
        window_config,
        Path(report_path),
    )
    return probabilities


def _validate_feature_range(features: pd.DataFrame) -> None:
    if not isinstance(features.index, pd.DatetimeIndex) or features.empty:
        raise ValueError("Market features must have a non-empty DatetimeIndex")
    if tuple(features.columns) != MARKET_FEATURES:
        raise ValueError(
            f"Market feature columns must be ordered as {MARKET_FEATURES}"
        )
    if not features.index.is_monotonic_increasing or features.index.has_duplicates:
        raise ValueError("Market feature dates must be unique and increasing")


def _window_dates(window_spec: object) -> list[tuple[str, str]]:
    if (
        isinstance(window_spec, list)
        and len(window_spec) == 2
        and all(isinstance(value, str) for value in window_spec)
    ):
        return [(window_spec[0], window_spec[1])]
    if isinstance(window_spec, list) and all(
        isinstance(item, list)
        and len(item) == 2
        and all(isinstance(value, str) for value in item)
        for item in window_spec
    ):
        return [(item[0], item[1]) for item in window_spec]
    raise ValueError(f"Invalid date window configuration: {window_spec!r}")


def _window_mask(
    index: pd.DatetimeIndex, window_spec: object
) -> pd.Series:
    mask = pd.Series(False, index=index)
    for start, end in _window_dates(window_spec):
        start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
        if start_date > end_date:
            raise ValueError(f"Window start {start} is after end {end}")
        mask |= (index >= start_date) & (index <= end_date)
    return mask


def _write_report(
    features: pd.DataFrame,
    probabilities: dict[str, pd.DataFrame],
    models: dict[str, RegimeModel],
    windows: dict,
    report_path: Path,
) -> None:
    sections = [
        "# Market Regime Report",
        "",
        "This pre-freeze report covers fit and validation windows only. "
        "No test-window regime assignments or statistics are included.",
        "",
    ]
    for model_label in ("final", "tuning"):
        model = models[model_label]
        probs = probabilities[model_label]
        report_windows = (
            ("calm", "covid", "rates")
            if model_label == "final"
            else ("calm", "covid", "rates", "val_2022")
        )
        sections.extend(
            [
                f"## {FIT_WINDOWS[model_label][1]} regime model",
                "",
                f"- Fit dates: {model.fit_start} through {model.fit_end}",
                "- Ordered mean z-scored `m_vol20`: "
                + ", ".join(f"{value:.4f}" for value in model.ordered_mean_m_vol20),
                "",
                "| Window | Dates | Days | Calm | Choppy | Crisis |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for window_name in report_windows:
            if window_name not in windows:
                continue
            mask = _window_mask(features.index, windows[window_name])
            window_probabilities = probs.loc[mask]
            if window_probabilities.empty:
                continue
            labels = window_probabilities.to_numpy().argmax(axis=1)
            shares = [
                float(np.mean(labels == regime_index))
                for regime_index in range(len(REGIME_NAMES))
            ]
            dates = ", ".join(
                f"{start}–{end}" for start, end in _window_dates(windows[window_name])
            )
            sections.append(
                f"| {window_name} | {dates} | {len(labels)} | "
                + " | ".join(f"{share:.1%}" for share in shares)
                + " |"
            )

        fit_mask = _window_mask(
            features.index, [model.fit_start, model.fit_end]
        )
        fit_features = features.loc[fit_mask]
        fit_probs = probs.loc[fit_mask]
        hard_labels = fit_probs.to_numpy().argmax(axis=1)
        sections.extend(
            [
                "",
                "### Mean market features by fitted regime",
                "",
                "| Regime | Days | m_vol20 | m_corr20 | m_dd |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for regime_index, regime_name in enumerate(REGIME_NAMES):
            regime_mask = hard_labels == regime_index
            means = fit_features.iloc[regime_mask].mean(axis=0)
            sections.append(
                f"| {regime_name.title()} | {int(regime_mask.sum())} | "
                + " | ".join(f"{means[name]:.6f}" for name in MARKET_FEATURES)
                + " |"
            )
        sections.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")


def _atomic_joblib_dump(model: RegimeModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".joblib", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(model, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".parquet", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_parquet(temporary_path, index=True)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    outputs = build_regime_models(
        args.features,
        args.config,
        args.artifacts_dir,
        args.processed_dir,
        args.report,
    )
    for label, output in outputs.items():
        print(
            f"{label}: generated {len(output)} causal regime probabilities to "
            f"{args.processed_dir / f'regime_probs_{label}.parquet'}"
        )
    print(f"Regime report: {args.report}")


if __name__ == "__main__":
    main()
