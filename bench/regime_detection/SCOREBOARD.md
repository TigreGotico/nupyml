# Scoreboard — regime_detection

**Goal:** Cluster timesteps into their generating regime; adjusted Rand index.

**Metric:** `adjusted_rand` (higher is better) · **QA floor:** `0.3`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_kmeans` | 0.7099 | 0.57 |
| 2 | `baseline_gmm` | 0.5595 | 0.55 |
| 3 | `baseline_threshold` | 0.3292 | 0.19 |

_Regenerate with `python bench/harness.py regime_detection`._
