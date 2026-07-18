# Scoreboard — volatility_forecast

**Goal:** Forecast the variance of a returns series; RMSE vs realised r^2 (lower is better).

**Metric:** `rmse` (lower is better) · **QA floor:** `3.6`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_garch` | 2.6227 | 0.37 |
| 2 | `baseline_ewma` | 2.6861 | 0.08 |
| 3 | `baseline_historical_var` | 2.8417 | 0.11 |

_Regenerate with `python bench/harness.py volatility_forecast`._
