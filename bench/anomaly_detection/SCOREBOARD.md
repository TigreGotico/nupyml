# Scoreboard — anomaly_detection

**Goal:** Rank points by how anomalous they are (unsupervised); scored by ROC-AUC against the hidden outlier labels.

**Metric:** `roc_auc` (higher is better) · **QA floor:** `0.85`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_isolation_forest` | 1.0000 | 13.80 |
| 2 | `baseline_knn` | 1.0000 | 3.14 |
| 3 | `baseline_hbos` | 1.0000 | 4.30 |
| 4 | `baseline_ecod` | 0.9841 | 5.79 |

_Regenerate with `python bench/harness.py anomaly_detection`._
