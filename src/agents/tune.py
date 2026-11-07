"""TU-03 lambda grid search and TU-04 validation comparison on val_2022."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agents.curriculum import load_config
from src.agents.train import STRATEGIES, save_checkpoint, train
from src.evaluation.backtest import degeneracy_guard, finalize_frame, greedy_pass, make_env
from src.evaluation.metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TUNING_DIR = PROJECT_ROOT / "results" / "tuning"
VALIDATION_DIR = PROJECT_ROOT / "results" / "validation" / "val_2022"
MAIN_STRATEGIES = ("tabular_q", "dqn", "qrdqn_a1.0", "qrdqn_a0.2")  # TU-04 simplicity order
SORTINO_TIE = 0.1


def run_and_score(task: dict[str, Any]) -> dict[str, Any]:
    """Train one tuning-pool run, then score a greedy pass on val_2022."""
    strategy, seed, steps = task["strategy"], task["seed"], task["steps"]
    lambda1, lambda2 = task["lambda1"], task["lambda2"]
    agent, log = train(
        strategy, seed, tuning=True, total_steps=steps, lambda1=lambda1, lambda2=lambda2
    )
    tag = task["tag"]
    run_dir = TUNING_DIR / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    name = f"{strategy}_seed{seed}"
    log.to_csv(run_dir / f"{name}.csv", index=False)
    save_checkpoint(
        agent,
        PROJECT_ROOT / "artifacts" / "models" / "tuning" / tag / f"{name}.pt",
        {"strategy": strategy, "seed": seed, "tuning": True, "steps": steps,
         "lambda1": lambda1, "lambda2": lambda2, "fit_window": "2012-2021"},
    )
    env = make_env("tuning", regime_enabled=STRATEGIES[strategy].regime, lambda1=lambda1, lambda2=lambda2)
    frame, infos = greedy_pass(agent, env, "val_2022")
    if task.get("write_validation_csv"):
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        finalize_frame(frame, "val_2022").to_csv(VALIDATION_DIR / f"{name}.csv", index=False)
    metrics = compute_metrics(frame["daily_return"], frame["turnover"], frame["cost_paid"])
    return {
        "tag": tag,
        "strategy": strategy,
        "seed": seed,
        "steps": steps,
        "lambda1": lambda1,
        "lambda2": lambda2,
        **metrics,
        **degeneracy_guard(infos),
    }


def run_tasks(tasks: list[dict[str, Any]], workers: int) -> pd.DataFrame:
    if workers <= 1:
        results = [run_and_score(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_and_score, tasks))
    return pd.DataFrame(results)


def select_lambda_cell(runs: pd.DataFrame) -> pd.DataFrame:
    """TU-03: per cell median Sortino over guard-passing seeds; best cell is flagged."""
    rows = []
    for cell, group in runs.groupby("cell", sort=True):
        passing = group[group["guard_passed"]]
        rows.append(
            {
                "cell": cell,
                "lambda1": group["lambda1"].iloc[0],
                "lambda2": group["lambda2"].iloc[0],
                "seeds_passing": len(passing),
                "median_sortino": passing["sortino"].median() if len(passing) else math.nan,
                "median_max_drawdown": passing["max_drawdown"].median() if len(passing) else math.nan,
            }
        )
    table = pd.DataFrame(rows)
    eligible = table.dropna(subset=["median_sortino"])
    table["selected"] = False
    if not eligible.empty:
        table.loc[eligible["median_sortino"].idxmax(), "selected"] = True
    return table


def select_headline(runs: pd.DataFrame) -> dict[str, Any]:
    """TU-04: guard, median Sortino, drawdown tie-break within 0.1, then simplicity."""
    passing = runs[runs["guard_passed"] & runs["strategy"].isin(MAIN_STRATEGIES)]
    if passing.empty:
        return {"headline": None, "reason": "no run passed the RW-04 degeneracy guard"}
    table = (
        passing.groupby("strategy")
        .agg(median_sortino=("sortino", "median"), median_max_drawdown=("max_drawdown", "median"),
             seeds_passing=("seed", "count"))
        .reset_index()
    )
    table["simplicity"] = table["strategy"].map(MAIN_STRATEGIES.index)
    table = table.sort_values("median_sortino", ascending=False).reset_index(drop=True)
    contenders = table.iloc[:2] if len(table) > 1 else table.iloc[:1]
    if len(contenders) == 2 and contenders["median_sortino"].iloc[0] - contenders["median_sortino"].iloc[1] < SORTINO_TIE:
        drawdown_gap = abs(contenders["median_max_drawdown"].iloc[0] - contenders["median_max_drawdown"].iloc[1])
        if drawdown_gap > 1e-12:
            winner = contenders.loc[contenders["median_max_drawdown"].idxmin()]
            rule = "top two within 0.1 Sortino; lower median max drawdown"
        else:
            winner = contenders.loc[contenders["simplicity"].idxmin()]
            rule = "top two tied on Sortino and drawdown; simpler model"
    else:
        winner = table.iloc[0]
        rule = "highest median Sortino"
    return {
        "headline": winner["strategy"],
        "rule_applied": rule,
        "ranking": table.drop(columns="simplicity").to_dict(orient="records"),
    }


def lambda_grid(steps: int, workers: int) -> None:
    config = load_config()
    tasks = [
        {"tag": f"lambda_C{cell}", "cell": cell, "strategy": "dqn", "seed": seed, "steps": steps,
         "lambda1": float(l1), "lambda2": float(l2)}
        for cell, (l1, l2) in enumerate(config["tuning"]["lambda_grid"])
        for seed in config["tuning"]["seeds"]
    ]
    runs = run_tasks(tasks, workers)
    runs["cell"] = [task["cell"] for task in tasks]
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(TUNING_DIR / "lambda_runs.csv", index=False)
    table = select_lambda_cell(runs)
    table.to_csv(TUNING_DIR / "lambda_grid.csv", index=False)
    print(runs.to_string(index=False))
    print(table.to_string(index=False))


def compare(steps: int, workers: int, seeds: list[int]) -> None:
    config = load_config()
    lambda1, lambda2 = config["reward"]["lambda1"], config["reward"]["lambda2"]
    if lambda1 is None or lambda2 is None:
        raise RuntimeError("set reward.lambda1/lambda2 (TU-03) before the validation comparison")
    tasks = [
        {"tag": "compare", "strategy": strategy, "seed": seed, "steps": steps,
         "lambda1": float(lambda1), "lambda2": float(lambda2), "write_validation_csv": True}
        for strategy in MAIN_STRATEGIES
        for seed in seeds
    ]
    runs = run_tasks(tasks, workers)
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(TUNING_DIR / "validation_comparison.csv", index=False)
    headline = select_headline(runs)
    headline.update({"steps": steps, "seeds": seeds, "lambda1": lambda1, "lambda2": lambda2})
    (TUNING_DIR / "headline.json").write_text(
        json.dumps(headline, indent=2, default=lambda value: None if value != value else value) + "\n",
        encoding="utf-8",
    )
    print(runs.to_string(index=False))
    print(json.dumps(headline, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    grid = sub.add_parser("lambda-grid")
    grid.add_argument("--steps", type=int, default=None)
    grid.add_argument("--workers", type=int, default=6)
    comp = sub.add_parser("compare")
    comp.add_argument("--steps", type=int, default=None)
    comp.add_argument("--workers", type=int, default=8)
    comp.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    config = load_config()
    if args.command == "lambda-grid":
        lambda_grid(args.steps or int(config["agents"]["tuning_steps"]), args.workers)
    else:
        compare(
            args.steps or int(config["agents"]["train_steps"]),
            args.workers,
            args.seeds or list(config["tuning"]["seeds"]),
        )


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
