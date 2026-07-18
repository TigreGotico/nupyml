# Scoreboard — ood_detection

**Goal:** Score points by how out-of-distribution they are; ROC-AUC.

**Metric:** `roc_auc` (higher is better) · **QA floor:** `0.85`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_pca_recon` | 0.9695 | 2.52 |
| 2 | `baseline_mahalanobis` | 0.9622 | 2.80 |
| 3 | `baseline_isolation_forest` | 0.8993 | 13.88 |

_Regenerate with `python bench/harness.py ood_detection`._
