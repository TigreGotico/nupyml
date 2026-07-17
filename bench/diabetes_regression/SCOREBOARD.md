# Scoreboard — diabetes_regression

**Goal:** Predict quantitative diabetes progression from ten baseline measurements.

**Metric:** `r2` (higher is better) · **QA floor:** `0.15`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_ridge` | 0.3652 | 0.49 |
| 2 | `baseline_gradient_boosting` | 0.3340 | 1.18 |
| 3 | `baseline_random_forest` | 0.2839 | 25.85 |
| 4 | `baseline_kernel_ridge` | 0.2089 | 0.64 |

_Regenerate with `python bench/harness.py diabetes_regression`._
