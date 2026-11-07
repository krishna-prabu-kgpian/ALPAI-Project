import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import (
    config_hash,
    degeneracy_guard,
    finalize_frame,
    greedy_pass,
    make_env,
    require_freeze,
)


class AlwaysHold:
    def greedy_action(self, observation):
        return 0


class Churn:
    def __init__(self):
        self.calls = 0

    def greedy_action(self, observation):
        self.calls += 1
        return 1 if self.calls % 2 else 6  # NVDA->AAPL then AAPL->NVDA


def test_test_windows_refuse_to_run_without_matching_freeze(tmp_path):
    frozen = tmp_path / "FROZEN"
    require_freeze("val_2022", frozen_path=frozen)
    with pytest.raises(RuntimeError, match="FROZEN does not exist"):
        require_freeze("test_war", frozen_path=frozen)
    frozen.write_text("not-the-hash\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        require_freeze("test_normal", frozen_path=frozen)
    frozen.write_text(config_hash() + "\n", encoding="utf-8")
    require_freeze("test_normal", frozen_path=frozen)


def test_greedy_pass_over_val_2022_writes_c01_rows():
    env = make_env("tuning", lambda1=0.0, lambda2=0.0)
    frame, infos = greedy_pass(AlwaysHold(), env, "val_2022")
    assert len(frame) == len(infos) > 240
    assert frame["date"].iloc[0] > "2022-01-01" and frame["date"].iloc[-1] <= "2022-12-31"
    assert frame["cost_paid"].sum() == 0.0
    guard = degeneracy_guard(infos)
    assert guard["hold_fraction"] == 1.0 and not guard["guard_passed"]

    final = finalize_frame(frame, "val_2022")
    assert list(final.columns[-4:]) == ["regime", "drawdown", "rolling_sharpe_21", "rolling_sortino_21"]
    assert len(final.columns) == 18
    assert final["rolling_sharpe_21"].iloc[:20].isna().all()
    values = np.concatenate(([1.0], frame["portfolio_value"].to_numpy()))
    np.testing.assert_allclose(np.cumprod(1 + frame["daily_return"]), values[1:], rtol=1e-12)


def test_guard_flags_churn():
    env = make_env("tuning", lambda1=0.0, lambda2=0.0)
    _, infos = greedy_pass(Churn(), env, "val_2022")
    guard = degeneracy_guard(infos)
    assert guard["hold_fraction"] == 0.0
    assert guard["mean_turnover"] == pytest.approx(0.2, abs=0.01)
    assert guard["guard_passed"]
    assert isinstance(pd.Timestamp(infos[0]["date"]), pd.Timestamp)
