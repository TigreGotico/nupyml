# Scoreboard — learning_to_rank

**Goal:** Rank items within each query by relevance; scored by mean NDCG@10 over the test queries.

**Metric:** `ndcg@10` (higher is better) · **QA floor:** `0.8`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_pointwise_ridge` | 0.9884 | 0.56 |
| 2 | `baseline_ranknet` | 0.9865 | 3.18 |
| 3 | `baseline_lambdamart` | 0.9000 | 2.66 |

_Regenerate with `python bench/harness.py learning_to_rank`._
