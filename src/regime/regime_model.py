"""Fit-window-only Gaussian mixture model for market regimes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.features.scaler import FeatureScaler

MARKET_FEATURES = ("m_vol20", "m_corr20", "m_dd")
REGIME_NAMES = ("calm", "choppy", "crisis")
REGIME_PROBABILITY_COLUMNS = ("p_calm", "p_choppy", "p_crisis")


@dataclass
class RegimeModel:
    """Market scaler and volatility-ordered diagonal Gaussian mixture."""

    scaler: FeatureScaler
    gmm: GaussianMixture
    component_order: np.ndarray
    fit_start: str
    fit_end: str
    ema_span: int = 5

    @classmethod
    def fit(
        cls,
        market_features: pd.DataFrame,
        fit_start: str,
        fit_end: str,
        *,
        n_components: int = 3,
        n_init: int = 5,
        random_state: int = 0,
        ema_span: int = 5,
    ) -> RegimeModel:
        _validate_market_features(market_features)
        if n_components != 3:
            raise ValueError("The regime specification requires exactly 3 components")
        if ema_span < 1:
            raise ValueError("ema_span must be a positive integer")
        scaler = FeatureScaler().fit(
            _as_scaler_features(market_features), fit_start, fit_end
        )
        fit_mask = (market_features.index >= pd.Timestamp(fit_start)) & (
            market_features.index <= pd.Timestamp(fit_end)
        )
        fit_data = _transform_with_scaler(market_features.loc[fit_mask], scaler)
        if len(fit_data) < n_components:
            raise ValueError("Fit window contains fewer rows than GMM components")
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            n_init=n_init,
            random_state=random_state,
        )
        gmm.fit(fit_data)
        component_order = np.argsort(gmm.means_[:, 0], kind="stable")
        return cls(
            scaler=scaler,
            gmm=gmm,
            component_order=component_order,
            fit_start=pd.Timestamp(fit_start).date().isoformat(),
            fit_end=pd.Timestamp(fit_end).date().isoformat(),
            ema_span=ema_span,
        )

    def posterior(self, market_features: pd.DataFrame) -> pd.DataFrame:
        """Return day-t-only posteriors, ordered Calm through Crisis."""
        _validate_market_features(market_features)
        transformed = _transform_with_scaler(market_features, self.scaler)
        probabilities = self.gmm.predict_proba(transformed)[:, self.component_order]
        result = pd.DataFrame(
            probabilities,
            index=market_features.index,
            columns=REGIME_PROBABILITY_COLUMNS,
        )
        result.columns.name = "regime_probability"
        return result

    def predict_proba(self, market_features: pd.DataFrame) -> pd.DataFrame:
        """Return causal EMA-smoothed posterior probabilities."""
        posterior = self.posterior(market_features)
        smoothed = posterior.ewm(
            span=self.ema_span, adjust=False, min_periods=1
        ).mean()
        smoothed = smoothed.div(smoothed.sum(axis=1), axis=0)
        smoothed.columns.name = "regime_probability"
        return smoothed

    @property
    def ordered_mean_m_vol20(self) -> np.ndarray:
        """GMM means for z-scored m_vol20, ordered Calm to Crisis."""
        return self.gmm.means_[self.component_order, 0].copy()


def _validate_market_features(features: pd.DataFrame) -> None:
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("Market features must be a non-empty DataFrame")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("Market feature index must be a DatetimeIndex")
    if features.index.hasnans or features.index.has_duplicates:
        raise ValueError("Market feature dates must not contain NaT or duplicates")
    if not features.index.is_monotonic_increasing:
        raise ValueError("Market feature dates must be strictly increasing")
    if tuple(features.columns) != MARKET_FEATURES:
        raise ValueError(
            f"Market feature columns must be ordered as {MARKET_FEATURES}"
        )
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Market features contain NaN or infinite values")


def _as_scaler_features(features: pd.DataFrame) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product(
        [["MARKET"], MARKET_FEATURES], names=["asset", "feature"]
    )
    values = features.copy()
    values.columns = columns
    return values


def _transform_with_scaler(
    features: pd.DataFrame, scaler: FeatureScaler
) -> np.ndarray:
    transformed = scaler.transform(_as_scaler_features(features))
    values = transformed.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Scaled market features contain NaN or infinite values")
    return values
