# Scoreboard — digits_classification

**Goal:** Classify 8x8 handwritten digits into the ten classes 0-9.

**Metric:** `accuracy` (higher is better) · **QA floor:** `0.9`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_svc` | 0.9833 | 0.57 |
| 2 | `baseline_random_forest` | 0.9777 | 7.15 |
| 3 | `baseline_mlp` | 0.9759 | 1.52 |
| 4 | `baseline_logistic` | 0.9740 | 0.37 |

_Regenerate with `python bench/harness.py digits_classification`._
