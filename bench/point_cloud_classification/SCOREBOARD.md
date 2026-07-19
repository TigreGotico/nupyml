# Scoreboard — point_cloud_classification

**Goal:** Classify unordered 3-D point clouds (sphere vs cube surface); accuracy.

**Metric:** `accuracy` (higher is better) · **QA floor:** `0.8`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_bag_of_points` | 1.0000 | 0.64 |
| 2 | `baseline_pointnet` | 0.8417 | 1.14 |

_Regenerate with `python bench/harness.py point_cloud_classification`._
