# Scoreboard — hierarchical_forecast

**Goal:** Forecast a 4-part hierarchy + its total coherently; RMSE (lower is better).

**Metric:** `rmse` (lower is better) · **QA floor:** `1.6`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_naive` | 0.4966 | 0.08 |
| 2 | `baseline_base_independent` | 0.5600 | 0.31 |
| 3 | `baseline_reconciled` | 0.5611 | 0.32 |

_Regenerate with `python bench/harness.py hierarchical_forecast`._
