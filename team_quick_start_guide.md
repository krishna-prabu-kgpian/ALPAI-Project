# 🚀 Team Quick Start Guide: Getting Started in 5 Minutes

Welcome to the **Risk-Sensitive RL Portfolio Allocation** project! We have less than 4 days, so here is the exact protocol for working together efficiently without wasting time or breaking the pipeline.

---

## 📚 1. How Our Files Work Together

* **`specs.md` (The Rules):** This is our **single source of truth**. It defines formulas, parameters, exact file paths, and test requirements. **If code contradicts `specs.md`, the code is wrong.**
* **`TODO.md` (The Plan):** This is our **step-by-step task checklist** (Steps 01–29). It tells you what to build next, what files to create, and what test checks must pass before moving on.

---

## ⚡ 2. The 3 Golden Rules (Read before touching code!)

1. **No Magic Numbers:** Never hardcode constants (like tickers, cost rates, hidden layers, or dates) in script files. Everything must read from `config/config.yaml`.
2. **Never Invent Rules:** If something isn't specified in `specs.md`, don't guess—add an entry under **Blockers** in `TODO.md` and ping the team.
3. **Green Tests Before Merge:** A step is only **DONE** when all its sub-task boxes are checked and the associated unit test (`T-01` through `T-17`) passes.

---

## 🛠️ 3. Workflow Protocol (Step-by-Step)

When starting or taking over a task, follow this loop:

```
[1. Pick a Step from TODO.md] ➡️ [2. Read its Spec Section] ➡️ [3. Write Code + Tests] ➡️ [4. Check off Boxes] ➡️ [5. Commit & Log]
```

1. **Pick your step:** Check `TODO.md` §3 (Status Board) and assign yourself as `Owner` for an unassigned `TODO` step.
2. **Read the corresponding spec section:** Look at the *"Read first"* note under the step details in `TODO.md` and read those specific sections in `specs.md`.
3. **Build & Test:** Write clean, modular code. Run `pytest` to make sure your assigned unit tests pass.
4. **Mark Complete:** Tick off the `- [ ]` checkboxes in `TODO.md` for your step and update the step status to `DONE`.
5. **Log & Commit:** Add a 1-line log entry at the bottom of `TODO.md` (Session Log) with your commit hash.

---

## 👥 4. Recommended Division of Labor (Phases 1 & 2)

To move fast, we can split into three parallel tracks right after **Step 02**:

| Teammate | Primary Assignment | Steps | Key Artifacts |
|---|---|---|---|
| **Member 1 (Data Lead)** | Data & Feature Engineering | Steps 03 ➔ 06 | `prices_clean.parquet`, `features_z_*.parquet`, Scalers |
| **Member 2 (Env Lead)** | Portfolio Engine & Gym Env | Steps 07 ➔ 10 | `portfolio_env.py`, Cost Model, Reward Engine |
| **Member 3 (Eval Lead)** | Metrics & Benchmarks | Steps 11 ➔ 12 | `metrics.py`, Buy & Hold, 1/N, MVO baseline scripts |

*Once Steps 03–12 are green, we re-group to train the RL agents sequentially (Steps 13–19).*

---

## ❓ Need Help or Found an Issue?

* **Spec disagreement or bug?** Follow the Change Process in `specs.md §19` (Open a PR with a Decision Log entry—never silently patch code).
* **Blocked on code?** Post your issue in the `Blockers` table at the bottom of `TODO.md`.

Let's keep it simple, stick to the checklist, and crush this deadline! 🚀