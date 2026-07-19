# Scoreboard — image_segmentation

**Goal:** Segment an image's pixels into regions; adjusted Rand index.

**Metric:** `adjusted_rand` (higher is better) · **QA floor:** `0.55`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_kmeans` | 1.0000 | 0.62 |
| 2 | `baseline_agglomerative` | 0.9954 | 0.59 |
| 3 | `baseline_gmm` | 0.9954 | 0.61 |

_Regenerate with `python bench/harness.py image_segmentation`._
