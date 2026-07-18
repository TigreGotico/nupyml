# Scoreboard — density_estimation

**Goal:** Fit a density and score held-out points by mean log-likelihood (higher is better).

**Metric:** `mean_log_likelihood` (higher is better) · **QA floor:** `-9.0`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_gmm` | -5.3229 | 1.40 |
| 2 | `baseline_kde` | -5.5954 | 1.18 |
| 3 | `reference_single_gaussian` | -6.4982 | 0.56 |

_Regenerate with `python bench/harness.py density_estimation`._
