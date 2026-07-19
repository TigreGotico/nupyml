# Scoreboard — timeseries_anomaly

**Goal:** Score timesteps by how anomalous they are; ROC-AUC vs injected anomalies.

**Metric:** `roc_auc` (higher is better) · **QA floor:** `0.6`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_isolation_forest` | 0.9295 | 3.16 |
| 2 | `baseline_zscore` | 0.6829 | 0.18 |
| 3 | `baseline_deviation` | 0.6558 | 0.21 |

_Regenerate with `python bench/harness.py timeseries_anomaly`._
