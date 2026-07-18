# Scoreboard — multilabel_emotions

**Goal:** Predict a set of correlated labels per sample; scored by macro-averaged F1 over the label columns.

**Metric:** `macro_f1_multilabel` (higher is better) · **QA floor:** `0.55`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_binary_relevance` | 0.8106 | 2.26 |
| 2 | `baseline_classifier_chain` | 0.8064 | 2.53 |
| 3 | `baseline_mlknn` | 0.6932 | 1.41 |

_Regenerate with `python bench/harness.py multilabel_emotions`._
