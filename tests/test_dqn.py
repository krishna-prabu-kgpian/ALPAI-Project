import numpy as np
import torch

from src.agents.dqn import DQNAgent
from src.agents.replay import ReplayBuffer


def test_replay_buffer_wraps_and_samples_stored_transitions():
    buffer = ReplayBuffer(3, 2, seed=0)
    for value in range(5):
        observation = np.full(2, value, dtype=np.float32)
        buffer.add(observation, value, float(value), observation + 1, False)
    assert len(buffer) == 3
    batch = buffer.sample(50)
    assert set(batch["actions"].tolist()) <= {2, 3, 4}
    np.testing.assert_array_equal(batch["observations"][:, 0], batch["rewards"])


def test_dqn_architecture_matches_spec():
    agent = DQNAgent(39, seed=0)
    shapes = [tuple(p.shape) for p in agent.online.parameters()]
    assert shapes == [(128, 39), (128,), (128, 128), (128,), (31, 128), (31,)]
    assert agent.target_update == 500


def test_dqn_toy_mdp_loss_decreases_and_learns_best_action():
    # Two states, three actions; action 2 pays +1, others pay 0; one-step episodes.
    agent = DQNAgent(2, n_actions=3, hidden=[32, 32], lr=1e-2, gamma=0.0, learn_start=32, target_update=50, seed=16)
    rng = np.random.default_rng(16)
    states = np.eye(2, dtype=np.float32)
    losses = []
    for _ in range(1500):
        state = states[rng.integers(2)]
        action = int(rng.integers(3))
        loss = agent.observe(state, action, float(action == 2), state, True)
        if loss is not None:
            losses.append(loss)
    early, late = np.mean(losses[:100]), np.mean(losses[-100:])
    assert late < 0.1 * early
    assert all(agent.greedy_action(state) == 2 for state in states)
    with torch.no_grad():
        q = agent.online(torch.as_tensor(states))
    np.testing.assert_allclose(q[:, 2].numpy(), 1.0, atol=0.1)


def test_epsilon_one_explores_and_epsilon_zero_is_greedy():
    agent = DQNAgent(4, seed=1)
    observation = np.zeros(4, dtype=np.float32)
    explored = {agent.act(observation, 1.0) for _ in range(500)}
    assert len(explored) > 20
    assert len({agent.act(observation, 0.0) for _ in range(20)}) == 1
