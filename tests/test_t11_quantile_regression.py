import numpy as np
import torch
from scipy.stats import norm

from src.agents.qrdqn import QRDQNAgent, quantile_huber_loss, quantile_midpoints

MU, SIGMA = 1.0, 4.0
N = 20


def test_t11_quantile_regression_recovers_normal_inverse_cdf():
    torch.manual_seed(11)
    taus = quantile_midpoints(N)
    theta = torch.zeros(1, N, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=0.05)
    for _ in range(3000):
        samples = MU + SIGMA * torch.randn(1, 256)
        loss = quantile_huber_loss(theta, samples, taus, kappa=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    expected = norm.ppf(taus.numpy(), loc=MU, scale=SIGMA)
    learned = theta.detach().numpy()[0]
    np.testing.assert_allclose(learned, expected, atol=0.15 * SIGMA)
    assert np.all(np.diff(learned) > 0)


def test_qrdqn_agent_learns_single_state_bandit_distribution():
    # Single state, gamma = 0: the action-0 quantiles must match N(mu, sigma).
    agent = QRDQNAgent(
        1,
        n_actions=2,
        n_quantiles=N,
        hidden=[32],
        lr=5e-3,
        gamma=0.0,
        learn_start=64,
        seed=11,
    )
    rng = np.random.default_rng(11)
    state = np.ones(1, dtype=np.float32)
    for _ in range(4000):
        agent.observe(state, 0, float(rng.normal(MU, SIGMA)), state, False)
    with torch.no_grad():
        learned = agent.quantiles(agent.online, torch.ones(1, 1))[0, 0].numpy()
    expected = norm.ppf(quantile_midpoints(N).numpy(), loc=MU, scale=SIGMA)
    np.testing.assert_allclose(np.sort(learned), expected, atol=0.25 * SIGMA)


def test_quantile_huber_loss_shape_is_batch_n_n():
    predicted = torch.zeros(5, N)
    target = torch.ones(5, N)
    loss = quantile_huber_loss(predicted, target, quantile_midpoints(N))
    # all u = 1 > 0: weight tau_i, Huber 0.5 -> sum_i tau_i * 0.5 = N/2 * 0.5
    assert loss.item() == np.float32(N / 2 * 0.5)
