# Scoreboard — link_prediction

**Goal:** Classify node pairs as linked / not from graph features; ROC-AUC.

**Metric:** `roc_auc` (higher is better) · **QA floor:** `0.75`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_logistic` | 0.8576 | 0.74 |
| 2 | `baseline_adamic_adar` | 0.8498 | 0.19 |
| 3 | `baseline_random_forest` | 0.8310 | 5.86 |

_Regenerate with `python bench/harness.py link_prediction`._
