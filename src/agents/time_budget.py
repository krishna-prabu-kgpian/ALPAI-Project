"""Step 15.5: wall-clock per 10,000 steps, plan projection and seed reproducibility check."""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import pandas as pd

from src.agents.curriculum import load_config
from src.agents.train import train

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_ROOT / "reports" / "time_budget.md"
MEASURED = ("tabular_q", "dqn", "qrdqn_a0.2")


def measure(strategy: str, steps: int) -> float:
    started = time.perf_counter()
    train(strategy, 0, tuning=True, total_steps=steps, lambda1=0.05, lambda2=0.002)
    return time.perf_counter() - started


def reproducible(strategy: str, steps: int) -> bool:
    _, first = train(strategy, 0, tuning=True, total_steps=steps, lambda1=0.05, lambda2=0.002)
    _, second = train(strategy, 0, tuning=True, total_steps=steps, lambda1=0.05, lambda2=0.002)
    return first.equals(second)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10_000)
    args = parser.parse_args()
    config = load_config()
    agents, tuning = config["agents"], config["tuning"]

    per_10k = {name: measure(name, args.steps) * 10_000 / args.steps for name in MEASURED}
    per_10k["qrdqn_a1.0"] = per_10k["qrdqn_a0.2"]
    per_10k["qrdqn_a0.2_noregime"] = per_10k["qrdqn_a0.2"]
    per_10k["qrdqn_a0.2_nocurr"] = per_10k["qrdqn_a0.2"]
    repro = {name: reproducible(name, 1_000) for name in MEASURED}

    lambda_runs = len(tuning["lambda_grid"]) * len(tuning["seeds"])
    lambda_seconds = lambda_runs * per_10k["dqn"] * agents["tuning_steps"] / 10_000
    compare_names = ("tabular_q", "dqn", "qrdqn_a1.0", "qrdqn_a0.2")
    compare_seconds = sum(per_10k[n] for n in compare_names) * len(tuning["seeds"]) * agents["train_steps"] / 10_000
    final_seconds = sum(per_10k.values()) * len(agents["seeds"]) * agents["train_steps"] / 10_000
    plan = pd.DataFrame(
        [
            ("Step 17 lambda grid", lambda_runs, agents["tuning_steps"], lambda_seconds),
            ("Step 20 validation comparison", len(compare_names) * len(tuning["seeds"]), agents["train_steps"], compare_seconds),
            ("Step 21 final training", len(per_10k) * len(agents["seeds"]), agents["train_steps"], final_seconds),
        ],
        columns=["phase", "runs", "steps_per_run", "serial_seconds"],
    )
    workers = max(1, (os.cpu_count() or 2) // 2)

    lines = [
        "# Time Budget (Step 15, TU-05)",
        "",
        f"Machine: {platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, "
        f"Python {platform.python_version()}, torch single-threaded per run, CPU only.",
        f"Measured on {args.steps:,} tuning-pool steps per strategy (learning starts after "
        f"{agents['learn_start']:,} steps).",
        "",
        "| Strategy | Seconds per 10,000 steps |",
        "|---|---|",
        *(f"| `{name}` | {per_10k[name]:.1f} |" for name in MEASURED),
        "",
        "QR-DQN variants and controls are assumed to cost the same as `qrdqn_a0.2`.",
        "",
        f"| Phase | Runs | Steps per run | Serial hours | Hours with {workers} parallel runs |",
        "|---|---|---|---|---|",
        *(
            f"| {row.phase} | {row.runs} | {row.steps_per_run:,} | {row.serial_seconds / 3600:.2f} | "
            f"{row.serial_seconds / 3600 / min(workers, row.runs):.2f} |"
            for row in plan.itertuples()
        ),
        "",
        "Reproducibility: 1,000-step training logs from two runs with seed 0 are "
        + ("identical for every measured strategy." if all(repro.values())
           else "NOT identical for " + ", ".join(n for n, ok in repro.items() if not ok) + "."),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
