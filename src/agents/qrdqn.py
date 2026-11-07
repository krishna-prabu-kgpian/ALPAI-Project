"""Quantile Regression DQN with CVaR action selection (M-04, DL-06)."""

from __future__ import annotations

import math
from typing import Any

import torch

from src.agents.dqn import NetworkAgent


def cvar_k(alpha: float, n_quantiles: int) -> int:
    """Number of lowest quantiles averaged by CVaR_alpha: max(1, ceil(round(alpha N, 8)))."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    return max(1, math.ceil(round(alpha * n_quantiles, 8)))


def quantile_midpoints(n_quantiles: int) -> torch.Tensor:
    """tau_hat_i = (2i - 1) / (2N) for i = 1..N."""
    return (2 * torch.arange(1, n_quantiles + 1, dtype=torch.float32) - 1) / (2 * n_quantiles)


def cvar_values(quantiles: torch.Tensor, alpha: float) -> torch.Tensor:
    """Mean of the k lowest quantiles along the last axis, after sorting ascending."""
    k = cvar_k(alpha, quantiles.shape[-1])
    return torch.sort(quantiles, dim=-1).values[..., :k].mean(dim=-1)


def quantile_huber_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile Huber loss over all N quantiles.

    ``predicted`` is [B, N] (theta_i), ``target`` is [B, N'] (target samples j).
    u_ij = target_j - theta_i; rho = |tau_i - 1{u<0}| L_kappa(u); sum over i,
    mean over j, mean over the batch.
    """
    u = target.unsqueeze(1) - predicted.unsqueeze(2)
    assert u.shape == (predicted.shape[0], predicted.shape[1], target.shape[1])
    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa, 0.5 * u.pow(2), kappa * (abs_u - 0.5 * kappa))
    weight = (taus.view(1, -1, 1) - (u.detach() < 0).float()).abs()
    return (weight * huber).sum(dim=1).mean(dim=1).mean()


class QRDQNAgent(NetworkAgent):
    """QR-DQN: 31 x N quantile head; CVaR_alpha drives acting and target action choice."""

    kind = "qrdqn"

    def __init__(
        self,
        obs_dim: int,
        *,
        n_quantiles: int = 20,
        alpha: float = 1.0,
        kappa: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.n_quantiles = int(n_quantiles)
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        cvar_k(self.alpha, self.n_quantiles)
        super().__init__(obs_dim, **kwargs)
        self.taus = quantile_midpoints(self.n_quantiles)

    @property
    def output_dim(self) -> int:
        return self.n_actions * self.n_quantiles

    def quantiles(self, network: torch.nn.Module, observations: torch.Tensor) -> torch.Tensor:
        return network(observations).view(-1, self.n_actions, self.n_quantiles)

    def action_values(self, observations: torch.Tensor) -> torch.Tensor:
        return cvar_values(self.quantiles(self.online, observations), self.alpha)

    def td_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch["actions"].shape[0]
        rows = torch.arange(batch_size)
        predicted = self.quantiles(self.online, batch["observations"])[rows, batch["actions"]]
        with torch.no_grad():
            next_quantiles = self.quantiles(self.target, batch["next_observations"])
            next_actions = cvar_values(next_quantiles, self.alpha).argmax(dim=1)
            target = batch["rewards"].unsqueeze(1) + self.gamma * (
                1.0 - batch["dones"].unsqueeze(1)
            ) * next_quantiles[rows, next_actions]
        return quantile_huber_loss(predicted, target, self.taus, self.kappa)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update({"n_quantiles": self.n_quantiles, "alpha": self.alpha})
        return state
