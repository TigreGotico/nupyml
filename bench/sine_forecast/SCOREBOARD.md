# Scoreboard — sine_forecast

**Goal:** Forecast the next 24 steps of a trended, seasonal, noisy series.

**Metric:** `rmse` (lower is better) · **QA floor:** `6.0`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_holt_winters` | 1.1988 | 1.53 |
| 2 | `baseline_arima` | 3.7564 | 2.67 |
| 3 | `baseline_naive_last` | 5.3014 | 0.27 |

_Regenerate with `python bench/harness.py sine_forecast`._
