# Scoreboard — fraud_imbalanced

**Goal:** Detect a rare fraud class (~8% positive) -- scored by macro-F1, not accuracy.

**Metric:** `macro_f1` (higher is better) · **QA floor:** `0.55`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_gradient_boosting` | 0.7645 | 2.95 |
| 2 | `baseline_random_forest` | 0.7291 | 3.81 |
| 3 | `baseline_smote_logistic` | 0.7264 | 0.50 |
| 4 | `baseline_logistic_plain` | 0.7231 | 0.41 |

_Regenerate with `python bench/harness.py fraud_imbalanced`._
