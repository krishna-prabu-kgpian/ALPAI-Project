"""Fit-window-only feature standardization and persistence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_raw.parquet"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STANDARD_DEVIATION_FLOOR = 1e-8
CLIP_BOUND = 5.0
SCHEMA_VERSION = 1


class FeatureScaler:
    """Z-score features using saved statistics from one explicit fit window."""

    def __init__(
        self,
        *,
        standard_deviation_floor: float = STANDARD_DEVIATION_FLOOR,
        clip_bound: float = CLIP_BOUND,
    ) -> None:
        if standard_deviation_floor <= 0 or clip_bound <= 0:
            raise ValueError("Scaler floor and clip bound must be positive")
        self.standard_deviation_floor = float(standard_deviation_floor)
        self.clip_bound = float(clip_bound)
        self.means: pd.Series | None = None
        self.standard_deviations: pd.Series | None = None
        self.fit_start: str | None = None
        self.fit_end: str | None = None
        self.fit_rows: int | None = None

    def fit(
        self,
        features: pd.DataFrame,
        fit_start: str,
        fit_end: str,
    ) -> FeatureScaler:
        """Fit population means/stds on rows in the inclusive date window only."""
        _validate_features(features)
        start = _parse_date(fit_start, "fit_start")
        end = _parse_date(fit_end, "fit_end")
        if start > end:
            raise ValueError("fit_start must not be after fit_end")

        fit_mask = (features.index >= start) & (features.index <= end)
        fit_features = features.loc[fit_mask]
        if fit_features.empty:
            raise ValueError(
                f"No feature rows fall within fit window {fit_start} to {fit_end}"
            )
        values = fit_features.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Fit-window features contain NaN or infinite values")

        self.means = fit_features.mean(axis=0)
        self.standard_deviations = fit_features.std(axis=0, ddof=0)
        self.fit_start = start.date().isoformat()
        self.fit_end = end.date().isoformat()
        self.fit_rows = len(fit_features)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform without fitting or modifying the saved fit statistics."""
        if self.means is None or self.standard_deviations is None:
            raise RuntimeError("Scaler must be fitted or loaded before transform")
        _validate_features(features)
        if not features.columns.equals(self.means.index):
            raise ValueError("Feature columns/order do not match the fitted scaler")
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("Features to transform contain NaN or infinite values")

        scale = self.standard_deviations.clip(lower=self.standard_deviation_floor)
        transformed = (features - self.means) / scale
        return transformed.clip(lower=-self.clip_bound, upper=self.clip_bound)

    def save(self, path: Path) -> None:
        """Persist fit window, ordered feature keys, means, and raw stds."""
        if (
            self.means is None
            or self.standard_deviations is None
            or self.fit_start is None
            or self.fit_end is None
            or self.fit_rows is None
        ):
            raise RuntimeError("Scaler must be fitted before it can be saved")

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "fit_window": {"start": self.fit_start, "end": self.fit_end},
            "fit_rows": self.fit_rows,
            "standard_deviation_floor": self.standard_deviation_floor,
            "clip_bound": self.clip_bound,
            "columns": [
                {"asset": str(asset), "feature": str(feature)}
                for asset, feature in self.means.index
            ],
            "means": [float(value) for value in self.means.to_numpy()],
            "standard_deviations": [
                float(value) for value in self.standard_deviations.to_numpy()
            ],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_json(metadata, Path(path))

    @classmethod
    def load(cls, path: Path) -> FeatureScaler:
        """Load saved statistics for evaluation; does not refit."""
        with Path(path).open(encoding="utf-8") as scaler_file:
            metadata = json.load(scaler_file)
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported scaler schema in {path}")
        try:
            columns = metadata["columns"]
            means = metadata["means"]
            standard_deviations = metadata["standard_deviations"]
            window = metadata["fit_window"]
            scaler = cls(
                standard_deviation_floor=metadata["standard_deviation_floor"],
                clip_bound=metadata["clip_bound"],
            )
            scaler.means = _series_from_json(columns, means)
            scaler.standard_deviations = _series_from_json(
                columns, standard_deviations
            )
            scaler.fit_start = _parse_date(window["start"], "fit_window.start").date().isoformat()
            scaler.fit_end = _parse_date(window["end"], "fit_window.end").date().isoformat()
            scaler.fit_rows = int(metadata["fit_rows"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid scaler artifact {path}: {error}") from error
        if (scaler.standard_deviations < 0).any():
            raise ValueError(f"Scaler artifact contains negative standard deviations: {path}")
        return scaler


def fit_configured_scalers(
    features_path: Path = DEFAULT_FEATURES_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> dict[str, pd.DataFrame]:
    """Fit the spec's final and tuning scalers and write full-series outputs."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    try:
        fit_windows = config["fit_windows"]
    except (TypeError, KeyError) as error:
        raise ValueError(f"Missing fit_windows configuration in {config_path}") from error
    features = pd.read_parquet(features_path)
    _validate_features(features)

    transformed_by_window: dict[str, pd.DataFrame] = {}
    for window_name, date_bounds in fit_windows.items():
        if not isinstance(date_bounds, list) or len(date_bounds) != 2:
            raise ValueError(f"Invalid fit window {window_name!r}: {date_bounds!r}")
        fit_start, fit_end = date_bounds
        scaler = FeatureScaler().fit(features, fit_start, fit_end)

        name_start = _parse_date(fit_start, "fit window start").year
        name_end = _parse_date(fit_end, "fit window end").year
        window_label = f"{name_start}-{name_end}"
        artifact_name = f"scaler_{window_label}.json"
        output_name = f"features_z_{window_label}.parquet"
        scaler_path = Path(artifacts_dir) / artifact_name
        output_path = Path(processed_dir) / output_name

        scaler.save(scaler_path)
        loaded_scaler = FeatureScaler.load(scaler_path)
        transformed = loaded_scaler.transform(features)
        _atomic_write_parquet(transformed, output_path)
        transformed_by_window[window_name] = transformed
    return transformed_by_window


def _validate_features(features: pd.DataFrame) -> None:
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("Features must be a non-empty pandas DataFrame")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("Feature index must be a DatetimeIndex")
    if features.index.hasnans or features.index.has_duplicates:
        raise ValueError("Feature index must not contain NaT or duplicate dates")
    if not features.index.is_monotonic_increasing:
        raise ValueError("Feature dates must be strictly increasing")
    if (
        not isinstance(features.columns, pd.MultiIndex)
        or features.columns.nlevels != 2
        or list(features.columns.names) != ["asset", "feature"]
    ):
        raise ValueError("Feature columns must be a (asset, feature) MultiIndex")
    if features.columns.has_duplicates:
        raise ValueError("Feature columns must be unique")


def _series_from_json(columns: list[dict], values: list[float]) -> pd.Series:
    if len(columns) != len(values):
        raise ValueError("feature column and scaler value counts differ")
    index = pd.MultiIndex.from_tuples(
        [(column["asset"], column["feature"]) for column in columns],
        names=["asset", "feature"],
    )
    return pd.Series(values, index=index, dtype=float)


def _parse_date(value: str, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a valid ISO date") from error
    if timestamp.tz is not None:
        raise ValueError(f"{label} must not include a timezone")
    return timestamp


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


def _atomic_write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(suffix=".json", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as scaler_file:
            json.dump(value, scaler_file, indent=2)
            scaler_file.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    args = parser.parse_args()

    outputs = fit_configured_scalers(
        args.features, args.config, args.artifacts_dir, args.processed_dir
    )
    for name, transformed in outputs.items():
        bounds = yaml.safe_load(args.config.read_text(encoding="utf-8"))["fit_windows"][name]
        window_label = (
            f"{_parse_date(bounds[0], 'fit window start').year}-"
            f"{_parse_date(bounds[1], 'fit window end').year}"
        )
        print(
            f"{name}: transformed {len(transformed)} rows to "
            f"{args.processed_dir / f'features_z_{window_label}.parquet'}"
        )


if __name__ == "__main__":
    main()
