# Scoreboard — node_classification

**Goal:** Classify graph nodes into communities from adjacency features; accuracy.

**Metric:** `accuracy` (higher is better) · **QA floor:** `0.7`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_logistic` | 1.0000 | 0.34 |
| 2 | `baseline_random_forest` | 0.8611 | 1.11 |
| 3 | `baseline_knn` | 0.8333 | 0.29 |

_Regenerate with `python bench/harness.py node_classification`._
