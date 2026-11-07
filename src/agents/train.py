"""Training runner: config, seeding, curriculum-driven episodes, logs and checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.agents.curriculum import CurriculumSampler, load_config
from src.agents.dqn import DQNAgent
from src.agents.qrdqn import QRDQNAgent
from src.agents.tabular_q import FIT_LABELS, TabularQAgent, fit_tabular_cuts
from src.evaluation.backtest import config_hash, make_env
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class StrategySpec:
    kind: str
    alpha: float | None = None
    regime: bool = True
    curriculum: bool = True


STRATEGIES = {
    "tabular_q": StrategySpec("tabular_q"),
    "dqn": StrategySpec("dqn"),
    "qrdqn_a1.0": StrategySpec("qrdqn", alpha=1.0),
    "qrdqn_a0.2": StrategySpec("qrdqn", alpha=0.2),
    "qrdqn_a0.2_noregime": StrategySpec("qrdqn", alpha=0.2, regime=False),
    "qrdqn_a0.2_nocurr": StrategySpec("qrdqn", alpha=0.2, curriculum=False),
}


def make_agent(strategy: str, obs_dim: int, config: dict[str, Any], seed: int, fit: str):
    spec = STRATEGIES[strategy]
    agents = config["agents"]
    if spec.kind == "tabular_q":
        return TabularQAgent(fit_tabular_cuts(fit), gamma=agents["gamma"], seed=seed)
    common = {
        "hidden": agents["hidden"],
        "lr": float(agents["lr"]),
        "gamma": float(agents["gamma"]),
        "batch_size": int(agents["batch"]),
        "replay_capacity": int(agents["replay"]),
        "learn_start": int(agents["learn_start"]),
        "target_update": int(agents["target_update"]),
        "seed": seed,
    }
    if spec.kind == "dqn":
        return DQNAgent(obs_dim, kappa=float(agents["kappa"]), **common)
    return QRDQNAgent(
        obs_dim,
        n_quantiles=int(agents["n_quantiles"]),
        alpha=float(spec.alpha),
        kappa=float(agents["kappa"]),
        **common,
    )


def train(
    strategy: str,
    seed: int,
    *,
    tuning: bool,
    total_steps: int,
    lambda1: float | None = None,
    lambda2: float | None = None,
    config: dict[str, Any] | None = None,
    log_every: int = 0,
) -> tuple[Any, pd.DataFrame]:
    """Train one strategy/seed and return the agent and its per-episode log."""
    config = load_config() if config is None else config
    spec = STRATEGIES[strategy]
    fit = "tuning" if tuning else "final"
    set_seed(seed)
    torch.set_num_threads(1)
    env = make_env(fit, regime_enabled=spec.regime, lambda1=lambda1, lambda2=lambda2, config=config)
    sampler = CurriculumSampler(
        env.feature_dates,
        config=config,
        total_steps=total_steps,
        tuning=tuning,
        curriculum_enabled=spec.curriculum,
        seed=seed,
    )
    agent = make_agent(strategy, env.observation_space.shape[0], config, seed, fit)

    rows: list[dict[str, Any]] = []
    episode = 0
    step = 0
    started = time.perf_counter()
    while step < total_steps:
        sample = sampler.sample(step)
        observation, _ = env.reset(seed=seed if episode == 0 else None, options=sample.options)
        episode_reward, losses, actions = 0.0, [], np.zeros(31, dtype=np.int64)
        epsilon = sampler.epsilon_at(step)
        truncated = False
        while not truncated and step < total_steps:
            epsilon = sampler.epsilon_at(step)
            action = agent.act(observation, epsilon)
            next_observation, reward, terminated, truncated, info = env.step(action)
            loss = agent.observe(observation, action, reward, next_observation, terminated)
            if loss is not None:
                losses.append(loss)
            episode_reward += reward
            actions[action] += 1
            observation = next_observation
            step += 1
        rows.append(
            {
                "global_step": step,
                "episode": episode,
                "stage": sample.stage,
                "window": sample.window,
                "epsilon": epsilon,
                "episode_reward": episode_reward,
                "mean_loss": float(np.mean(losses)) if losses else np.nan,
                "final_value": info["V"],
                "hold_fraction": actions[0] / max(1, actions.sum()),
            }
        )
        episode += 1
        if log_every and episode % log_every == 0:
            print(
                f"[{strategy} s{seed}] step {step}/{total_steps} "
                f"eps {epsilon:.3f} reward {episode_reward:.2f} "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    return agent, pd.DataFrame(rows)


def save_checkpoint(agent: Any, path: Path, metadata: dict[str, Any]) -> None:
    """M-06: model state plus the config hash and run metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": {**metadata, "config_hash": config_hash()}, "agent": agent.state_dict()}, path)


def load_checkpoint(path: Path, config: dict[str, Any] | None = None):
    """Rebuild an agent from a checkpoint written by save_checkpoint."""
    config = load_config() if config is None else config
    payload = torch.load(path, weights_only=False)
    metadata, state = payload["metadata"], payload["agent"]
    fit = "tuning" if metadata.get("tuning") else "final"
    agent = make_agent(metadata["strategy"], state.get("obs_dim", 39), config, metadata["seed"], fit)
    agent.load_state_dict(state)
    return agent, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--lambda1", type=float, default=None)
    parser.add_argument("--lambda2", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config()
    steps = args.steps or int(
        config["agents"]["tuning_steps" if args.tuning else "train_steps"]
    )
    agent, log = train(
        args.strategy,
        args.seed,
        tuning=args.tuning,
        total_steps=steps,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        config=config,
        log_every=50,
    )
    out_dir = args.out_dir or (
        PROJECT_ROOT / "results" / ("tuning/runs" if args.tuning else "training")
    )
    name = f"{args.strategy}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.to_csv(out_dir / f"{name}.csv", index=False)
    model_dir = PROJECT_ROOT / "artifacts" / "models" / ("tuning" if args.tuning else "")
    save_checkpoint(
        agent,
        model_dir / f"{name}.pt",
        {
            "strategy": args.strategy,
            "seed": args.seed,
            "tuning": args.tuning,
            "steps": steps,
            "lambda1": args.lambda1,
            "lambda2": args.lambda2,
            "fit_window": FIT_LABELS["tuning" if args.tuning else "final"],
        },
    )
    print(json.dumps({"run": name, "episodes": len(log)}))


if __name__ == "__main__":
    main()
