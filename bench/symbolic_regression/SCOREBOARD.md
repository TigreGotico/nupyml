# Scoreboard — symbolic_regression

**Goal:** Recover a nonlinear formula y=f(x); scored by held-out R^2.

**Metric:** `r2` (higher is better) · **QA floor:** `0.8`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_poly_ridge` | 0.9997 | 1.64 |
| 2 | `baseline_gbdt` | 0.9482 | 2.81 |

_Regenerate with `python bench/harness.py symbolic_regression`._
