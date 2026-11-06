"""Episode-level policy diagnostics from environment info records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def summarize_episode(infos: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Summarize cash allocation, action entropy, turnover, and HOLD frequency."""
    if not infos:
        raise ValueError("infos must contain at least one environment step")

    cash_weights = np.asarray([info["weights"][5] for info in infos], dtype=np.float64)
    turnovers = np.asarray([info["turnover"] for info in infos], dtype=np.float64)
    action_ids = np.asarray([info["action_id"] for info in infos], dtype=np.int64)
    if (
        not np.isfinite(cash_weights).all()
        or not np.isfinite(turnovers).all()
        or (cash_weights < 0).any()
        or (cash_weights > 1).any()
        or (turnovers < 0).any()
    ):
        raise ValueError("info records contain invalid weights or turnover")

    _, action_counts = np.unique(action_ids, return_counts=True)
    action_probabilities = action_counts / action_counts.sum()
    entropy = float(-np.sum(action_probabilities * np.log(action_probabilities)))
    return {
        "mean_cash": float(np.mean(cash_weights)),
        "action_entropy": entropy,
        "mean_turnover": float(np.mean(turnovers)),
        "hold_fraction": float(np.mean(action_ids == 0)),
    }