# Scoreboard — multi_output_regression

**Goal:** Predict a 3-dimensional target; scored by average R^2 over the outputs.

**Metric:** `avg_r2` (higher is better) · **QA floor:** `0.65`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_ridge` | 0.8174 | 1.75 |
| 2 | `baseline_multitask_enet` | 0.8171 | 1.31 |
| 3 | `baseline_rf` | 0.7090 | 134.41 |

_Regenerate with `python bench/harness.py multi_output_regression`._
