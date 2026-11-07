# Time Budget (Step 15, TU-05)

Machine: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, 22 logical CPUs, Python 3.12.7, torch single-threaded per run, CPU only.
Measured on 10,000 tuning-pool steps per strategy (learning starts after 1,000 steps).

| Strategy | Seconds per 10,000 steps |
|---|---|
| `tabular_q` | 11.0 |
| `dqn` | 61.9 |
| `qrdqn_a0.2` | 94.7 |

QR-DQN variants and controls are assumed to cost the same as `qrdqn_a0.2`.

| Phase | Runs | Steps per run | Serial hours | Hours with 11 parallel runs |
|---|---|---|---|---|
| Step 17 lambda grid | 6 | 75,000 | 0.77 | 0.13 |
| Step 20 validation comparison | 8 | 150,000 | 2.19 | 0.27 |
| Step 21 final training | 18 | 150,000 | 5.65 | 0.51 |

Reproducibility: 1,000-step training logs from two runs with seed 0 are identical for every measured strategy.
