# Scoreboard — gp_regression

**Goal:** Regress a smooth multi-frequency signal; scored by held-out R^2.

**Metric:** `r2` (higher is better) · **QA floor:** `0.85`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_exact_gp` | 0.9951 | 0.71 |
| 2 | `baseline_sparse_gp` | 0.9951 | 1.03 |
| 3 | `baseline_knn` | 0.9941 | 0.48 |

_Regenerate with `python bench/harness.py gp_regression`._
