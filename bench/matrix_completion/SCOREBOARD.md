# Scoreboard — matrix_completion

**Goal:** Fill held-out cells of a low-rank matrix from observed ones; RMSE.

**Metric:** `rmse` (lower is better) · **QA floor:** `1.0`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_matrix_factorization` | 0.8394 | 4.98 |
| 2 | `reference_row_mean` | 1.8833 | 0.35 |

_Regenerate with `python bench/harness.py matrix_completion`._
