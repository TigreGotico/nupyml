# Scoreboard — multivariate_forecast

**Goal:** Jointly forecast 3 coupled series; RMSE over all series (lower is better).

**Metric:** `rmse` (lower is better) · **QA floor:** `1.2`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_var` | 0.3725 | 0.32 |
| 2 | `baseline_per_series_ar` | 0.3770 | 0.32 |
| 3 | `baseline_naive_last` | 0.4920 | 0.07 |

_Regenerate with `python bench/harness.py multivariate_forecast`._
