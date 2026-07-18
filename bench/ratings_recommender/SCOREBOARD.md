# Scoreboard — ratings_recommender

**Goal:** Predict held-out (user, item) ratings from a low-rank matrix; scored by RMSE (lower is better).

**Metric:** `rmse` (lower is better) · **QA floor:** `0.85`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_matrix_factorization` | 0.5572 | 8.12 |
| 2 | `baseline_svdpp` | 0.5647 | 16.31 |
| 3 | `reference_item_cf` | 1.4532 | 1.68 |
| 4 | `reference_global_mean` | 1.9856 | 0.46 |

_Regenerate with `python bench/harness.py ratings_recommender`._
