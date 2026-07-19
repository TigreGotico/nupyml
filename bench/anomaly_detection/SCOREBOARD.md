# Scoreboard — anomaly_detection

**Goal:** Rank points by how anomalous they are (unsupervised); scored by ROC-AUC against the hidden outlier labels.

**Metric:** `roc_auc` (higher is better) · **QA floor:** `0.85`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_isolation_forest` | 1.0000 | 5.64 |
| 2 | `baseline_knn` | 1.0000 | 1.18 |
| 3 | `baseline_hbos` | 1.0000 | 1.27 |
| 4 | `baseline_ecod` | 0.9841 | 1.30 |

_Regenerate with `python bench/harness.py anomaly_detection`._
