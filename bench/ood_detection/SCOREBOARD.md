# Scoreboard — ood_detection

**Goal:** Score points by how out-of-distribution they are; ROC-AUC.

**Metric:** `roc_auc` (higher is better) · **QA floor:** `0.85`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_pca_recon` | 0.9695 | 1.00 |
| 2 | `baseline_mahalanobis` | 0.9622 | 0.99 |
| 3 | `baseline_isolation_forest` | 0.8993 | 5.13 |

_Regenerate with `python bench/harness.py ood_detection`._
