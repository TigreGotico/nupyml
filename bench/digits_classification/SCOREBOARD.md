# Scoreboard — digits_classification

**Goal:** Classify 8x8 handwritten digits into the ten classes 0-9.

**Metric:** `accuracy` (higher is better) · **QA floor:** `0.9`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_svc` | 0.9833 | 1.92 |
| 2 | `baseline_random_forest` | 0.9777 | 34.57 |
| 3 | `baseline_mlp` | 0.9759 | 6.81 |
| 4 | `baseline_logistic` | 0.9740 | 2.64 |

_Regenerate with `python bench/harness.py digits_classification`._
