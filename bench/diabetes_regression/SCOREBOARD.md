# Scoreboard — diabetes_regression

**Goal:** Predict quantitative diabetes progression from ten baseline measurements.

**Metric:** `r2` (higher is better) · **QA floor:** `0.15`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_ridge` | 0.3652 | 0.33 |
| 2 | `baseline_gradient_boosting` | 0.3340 | 0.66 |
| 3 | `baseline_random_forest` | 0.2839 | 16.63 |
| 4 | `baseline_kernel_ridge` | 0.2089 | 0.33 |

_Regenerate with `python bench/harness.py diabetes_regression`._
