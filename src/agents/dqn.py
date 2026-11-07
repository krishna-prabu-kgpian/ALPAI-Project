"""Standard Deep Q-Network (M-03) and the shared value-network agent machinery."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F

from src.agents.replay import ReplayBuffer

N_ACTIONS = 31


def build_mlp(input_dim: int, hidden: list[int], output_dim: int) -> nn.Sequential:
    """MLP with ReLU hidden layers (M-01)."""
    layers: list[nn.Module] = []
    width = input_dim
    for size in hidden:
        layers.extend([nn.Linear(width, size), nn.ReLU()])
        width = size
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class NetworkAgent:
    """Replay, epsilon-greedy acting and hard target updates shared by DQN and QR-DQN."""

    kind = "base"

    def __init__(
        self,
        obs_dim: int,
        *,
        n_actions: int = N_ACTIONS,
        hidden: list[int] | tuple[int, ...] = (128, 128),
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        replay_capacity: int = 100_000,
        learn_start: int = 1_000,
        target_update: int = 500,
        grad_clip: float = 10.0,
        seed: int | None = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = list(hidden)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.learn_start = int(learn_start)
        self.target_update = int(target_update)
        self.grad_clip = float(grad_clip)
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
        self.online = build_mlp(self.obs_dim, self.hidden, self.output_dim)
        self.target = build_mlp(self.obs_dim, self.hidden, self.output_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.replay = ReplayBuffer(replay_capacity, self.obs_dim, seed=seed)
        self.env_steps = 0
        self.gradient_steps = 0

    @property
    def output_dim(self) -> int:
        return self.n_actions

    def action_values(self, observations: torch.Tensor) -> torch.Tensor:
        """Scalar per-action values used for acting, shape [B, n_actions]."""
        raise NotImplementedError

    def td_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def greedy_action(self, observation: NDArray[np.float32]) -> int:
        with torch.no_grad():
            values = self.action_values(torch.as_tensor(observation).unsqueeze(0))
        return int(torch.argmax(values, dim=1).item())

    def act(self, observation: NDArray[np.float32], epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(observation)

    def observe(
        self,
        observation: NDArray[np.float32],
        action: int,
        reward: float,
        next_observation: NDArray[np.float32],
        done: bool,
    ) -> float | None:
        """Store a transition and take one gradient step once learning has started."""
        self.replay.add(observation, action, reward, next_observation, done)
        self.env_steps += 1
        if self.env_steps < self.learn_start:
            return None
        return self.learn()

    def learn(self) -> float:
        sample = self.replay.sample(self.batch_size)
        batch = {key: torch.as_tensor(value) for key, value in sample.items()}
        loss = self.td_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "obs_dim": self.obs_dim,
            "hidden": self.hidden,
            "online": self.online.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["online"])


class DQNAgent(NetworkAgent):
    """Plain DQN: Huber loss on r + gamma (1 - done) max_a' Q_target(s', a')."""

    kind = "dqn"

    def __init__(self, obs_dim: int, *, kappa: float = 1.0, **kwargs: Any) -> None:
        self.kappa = float(kappa)
        super().__init__(obs_dim, **kwargs)

    def action_values(self, observations: torch.Tensor) -> torch.Tensor:
        return self.online(observations)

    def td_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        q_values = self.online(batch["observations"])
        chosen = q_values.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_max = self.target(batch["next_observations"]).max(dim=1).values
            targets = batch["rewards"] + self.gamma * (1.0 - batch["dones"]) * next_max
        return F.huber_loss(chosen, targets, delta=self.kappa)
