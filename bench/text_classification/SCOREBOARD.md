# Scoreboard — text_classification

**Goal:** Classify short text documents into one of three topics; scored by accuracy.

**Metric:** `accuracy` (higher is better) · **QA floor:** `0.75`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_tfidf_logreg` | 0.8667 | 0.84 |
| 2 | `baseline_count_nb` | 0.8593 | 0.38 |

_Regenerate with `python bench/harness.py text_classification`._
