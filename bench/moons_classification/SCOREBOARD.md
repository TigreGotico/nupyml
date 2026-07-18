# Scoreboard — moons_classification

**Goal:** Separate two interleaving half-moons (nonlinear binary classification).

**Metric:** `accuracy` (higher is better) · **QA floor:** `0.9`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_mlp` | 0.9700 | 6.17 |
| 2 | `baseline_knn` | 0.9667 | 1.07 |
| 3 | `baseline_random_forest` | 0.9667 | 4.95 |
| 4 | `baseline_svc_rbf` | 0.9667 | 1.61 |

_Regenerate with `python bench/harness.py moons_classification`._
