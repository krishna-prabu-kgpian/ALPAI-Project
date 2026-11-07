# Market Regime Report

This pre-freeze report covers fit and validation windows only. No test-window regime assignments or statistics are included.

## 2012-2023 regime model

- Fit dates: 2012-01-01 through 2023-12-31
- Ordered mean z-scored `m_vol20`: -0.5969, 0.1452, 1.5210

| Window | Dates | Days | Calm | Choppy | Crisis |
|---|---|---:|---:|---:|---:|
| calm | 2012-01-01–2019-12-31, 2021-01-01–2021-12-31 | 2215 | 65.5% | 28.1% | 6.4% |
| covid | 2020-01-01–2020-12-31 | 253 | 24.9% | 39.5% | 35.6% |
| rates | 2022-01-01–2023-12-31 | 501 | 20.8% | 20.4% | 58.9% |

### Mean market features by fitted regime

| Regime | Days | m_vol20 | m_corr20 | m_dd |
|---|---:|---:|---:|---:|
| Calm | 1618 | 0.158903 | 0.366468 | 0.021372 |
| Choppy | 824 | 0.244944 | 0.635993 | 0.053207 |
| Crisis | 527 | 0.415847 | 0.763642 | 0.211080 |

## 2012-2021 regime model

- Fit dates: 2012-01-01 through 2021-12-31
- Ordered mean z-scored `m_vol20`: -0.4981, 0.3008, 2.0179

| Window | Dates | Days | Calm | Choppy | Crisis |
|---|---|---:|---:|---:|---:|
| calm | 2012-01-01–2019-12-31, 2021-01-01–2021-12-31 | 2215 | 64.5% | 29.6% | 5.9% |
| covid | 2020-01-01–2020-12-31 | 253 | 24.5% | 42.7% | 32.8% |
| rates | 2022-01-01–2023-12-31 | 501 | 19.0% | 24.6% | 56.5% |
| val_2022 | 2022-01-01–2022-12-31 | 251 | 0.0% | 11.2% | 88.8% |

### Mean market features by fitted regime

| Regime | Days | m_vol20 | m_corr20 | m_dd |
|---|---:|---:|---:|---:|
| Calm | 1491 | 0.155012 | 0.358280 | 0.021586 |
| Choppy | 764 | 0.245352 | 0.637019 | 0.054975 |
| Crisis | 213 | 0.466181 | 0.787744 | 0.150932 |

## Curriculum window difficulty

Daily-rebalanced equal-weight basket maximum drawdown, computed separately within each configured training segment. The calm value is the maximum across its two segments.

| Window | Maximum drawdown |
|---|---:|
| calm | 33.13% |
| covid | 28.44% |
| rates | 40.89% |

The observed severity order is rates > calm > covid, which differs from the prescribed curriculum order calm → covid → rates. Per CU-02/OQ-09, the training schedule remains unchanged pending a team decision; windows are not silently reordered.
