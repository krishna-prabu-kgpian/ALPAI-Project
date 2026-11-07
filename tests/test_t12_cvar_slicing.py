import torch

from src.agents.qrdqn import QRDQNAgent, cvar_k, cvar_values, quantile_midpoints


def test_t12_cvar_slicing_sorts_and_uses_k_lowest_quantiles():
    assert cvar_k(1.0, 20) == 20
    assert cvar_k(0.2, 20) == 4
    assert cvar_k(0.01, 20) == 1

    ordered = torch.arange(20, dtype=torch.float32)
    shuffled = ordered[torch.randperm(20, generator=torch.Generator().manual_seed(12))]
    assert not torch.equal(shuffled, ordered)
    # alpha = 0.2 averages the four lowest values 0, 1, 2, 3 even when unsorted
    assert cvar_values(shuffled, 0.2).item() == 1.5
    # alpha = 1.0 is the plain mean
    assert cvar_values(shuffled, 1.0).item() == ordered.mean().item()


def test_cvar_values_slice_last_axis_per_action():
    quantiles = torch.stack([torch.linspace(-1, 1, 20), torch.linspace(-3, 5, 20).flip(0)])
    values = cvar_values(quantiles.unsqueeze(0), 0.2)
    assert values.shape == (1, 2)
    expected = torch.sort(quantiles, dim=-1).values[:, :4].mean(dim=-1)
    torch.testing.assert_close(values[0], expected)


def test_quantile_midpoints_and_risk_averse_action_choice():
    torch.testing.assert_close(quantile_midpoints(20)[[0, 1, -1]], torch.tensor([0.025, 0.075, 0.975]))
    agent = QRDQNAgent(3, n_actions=2, n_quantiles=20, alpha=0.2, hidden=[8], seed=0)
    safe = torch.full((20,), 0.5)
    risky = torch.linspace(-4.0, 6.0, 20)  # higher mean, worse lower tail
    assert risky.mean() > safe.mean()
    stacked = torch.stack([safe, risky]).unsqueeze(0)
    assert int(cvar_values(stacked, agent.alpha).argmax(dim=1)) == 0
    assert int(cvar_values(stacked, 1.0).argmax(dim=1)) == 1
