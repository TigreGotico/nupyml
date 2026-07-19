# Scoreboard — intermittent_forecast

**Goal:** Forecast a sparse intermittent-demand series; RMSE (lower is better).

**Metric:** `rmse` (lower is better) · **QA floor:** `2.1`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_mean_rate` | 1.9870 | 0.19 |
| 2 | `baseline_moving_average` | 1.9875 | 0.13 |
| 3 | `baseline_croston` | 1.9912 | 0.71 |
| 4 | `reference_naive_last` | 2.2472 | 0.21 |

_Regenerate with `python bench/harness.py intermittent_forecast`._
