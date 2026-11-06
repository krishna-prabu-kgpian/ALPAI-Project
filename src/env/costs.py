"""Transaction fees and optional volatility-scaled slippage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

SlippageMode = Literal["none", "vol_scaled"]


@dataclass(frozen=True)
class CostModel:
    """Cost rates applied to trades in the five risky assets."""

    cost_rate: float = 0.001
    slippage_mode: SlippageMode = "none"
    slippage_s0_bps: float = 2.0
    median_fit_volatility: ArrayLike | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.cost_rate) or self.cost_rate < 0:
            raise ValueError("cost_rate must be finite and non-negative")
        if self.slippage_mode not in ("none", "vol_scaled"):
            raise ValueError("slippage_mode must be 'none' or 'vol_scaled'")
        if not np.isfinite(self.slippage_s0_bps) or self.slippage_s0_bps < 0:
            raise ValueError("slippage_s0_bps must be finite and non-negative")
        if self.slippage_mode == "vol_scaled":
            if self.median_fit_volatility is None:
                raise ValueError(
                    "median_fit_volatility is required when slippage_mode='vol_scaled'"
                )
            medians = np.asarray(self.median_fit_volatility, dtype=float)
            if (
                medians.shape != (5,)
                or not np.isfinite(medians).all()
                or (medians <= 0).any()
            ):
                raise ValueError(
                    "median_fit_volatility must contain five finite positive values"
                )

    def slippage_bps(self, volatility: ArrayLike | None = None) -> NDArray[np.float64]:
        """Return per-asset slippage bps, or zero when slippage is disabled."""
        if self.slippage_mode == "none":
            return np.zeros(5, dtype=float)
        if volatility is None:
            raise ValueError("volatility is required when slippage is enabled")
        current = np.asarray(volatility, dtype=float)
        medians = np.asarray(self.median_fit_volatility, dtype=float)
        if (
            current.shape != (5,)
            or not np.isfinite(current).all()
            or (current < 0).any()
        ):
            raise ValueError("volatility must contain five finite non-negative values")
        return self.slippage_s0_bps * np.clip(current / medians, 1.0, 5.0)

    def rates(self, volatility: ArrayLike | None = None) -> NDArray[np.float64]:
        """Return transaction cost plus slippage rate for each risky asset."""
        return self.cost_rate + self.slippage_bps(volatility) / 10_000.0
