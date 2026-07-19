# Scoreboard — survival_risk

**Goal:** Rank subjects by risk from censored survival data; scored by the concordance index.

**Metric:** `concordance_index` (higher is better) · **QA floor:** `0.7`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_coxph` | 0.8100 | 0.67 |
| 2 | `baseline_weibull_aft` | 0.8094 | 0.64 |
| 3 | `baseline_rsf` | 0.7603 | 5.80 |

_Regenerate with `python bench/harness.py survival_risk`._
