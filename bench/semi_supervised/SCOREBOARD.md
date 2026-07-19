# Scoreboard — semi_supervised

**Goal:** Classify with only ~8% of training labels (rest are -1); use the unlabeled points. Scored by accuracy.

**Metric:** `accuracy` (higher is better) · **QA floor:** `0.8`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_self_training` | 0.9533 | 1.85 |
| 2 | `reference_labeled_only` | 0.9400 | 0.60 |
| 3 | `baseline_label_spreading` | 0.9367 | 0.53 |

_Regenerate with `python bench/harness.py semi_supervised`._
