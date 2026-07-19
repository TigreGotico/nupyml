# Scoreboard — ts_classification

**Goal:** Classify each 1-D time series by its generating pattern (phase/location randomised); scored by accuracy.

**Metric:** `accuracy` (higher is better) · **QA floor:** `0.8`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_flatten_svc` | 0.9633 | 0.58 |
| 2 | `baseline_features_rf` | 0.9450 | 1.50 |
| 3 | `baseline_rocket` | 0.9083 | 10.03 |
| 4 | `baseline_dtw_knn` | 0.8440 | 23.97 |

_Regenerate with `python bench/harness.py ts_classification`._
