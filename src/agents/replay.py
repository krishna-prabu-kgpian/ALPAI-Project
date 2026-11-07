"""Uniform experience replay buffer (M-01)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ReplayBuffer:
    """Fixed-capacity ring buffer of (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int, obs_dim: int, seed: int | None = None) -> None:
        if capacity < 1 or obs_dim < 1:
            raise ValueError("capacity and obs_dim must be positive")
        self.capacity = int(capacity)
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self._next = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: NDArray[np.float32],
        action: int,
        reward: float,
        next_observation: NDArray[np.float32],
        done: bool,
    ) -> None:
        slot = self._next
        self.observations[slot] = observation
        self.actions[slot] = action
        self.rewards[slot] = reward
        self.next_observations[slot] = next_observation
        self.dones[slot] = float(done)
        self._next = (slot + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, NDArray]:
        """Draw ``batch_size`` transitions uniformly with replacement."""
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        indices = self.rng.integers(0, self._size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
        }
