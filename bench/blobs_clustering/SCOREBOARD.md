# Scoreboard — blobs_clustering

**Goal:** Cluster points into their generating blobs (unsupervised); scored by ARI.

**Metric:** `adjusted_rand_index` (higher is better) · **QA floor:** `0.7`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_agglomerative` | 1.0000 | 0.62 |
| 2 | `baseline_gaussian_mixture` | 1.0000 | 0.60 |
| 3 | `baseline_kmeans` | 1.0000 | 0.56 |

_Regenerate with `python bench/harness.py blobs_clustering`._
