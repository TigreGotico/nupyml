# Scoreboard — keypoint_matching

**Goal:** Classify shapes (square/disk/ring) at random position & scale; accuracy.

**Metric:** `accuracy` (higher is better) · **QA floor:** `0.8`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_hog_lbp_rf` | 0.8815 | 17.95 |
| 2 | `baseline_pixels_svc` | 0.8519 | 0.56 |

_Regenerate with `python bench/harness.py keypoint_matching`._
