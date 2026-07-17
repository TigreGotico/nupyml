# Scoreboard — sine_forecast

**Goal:** Forecast the next 24 steps of a trended, seasonal, noisy series.

**Metric:** `rmse` (lower is better) · **QA floor:** `6.0`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_holt_winters` | 1.1988 | 0.55 |
| 2 | `baseline_arima` | 3.7564 | 0.65 |
| 3 | `baseline_naive_last` | 5.3014 | 0.18 |

_Regenerate with `python bench/harness.py sine_forecast`._
