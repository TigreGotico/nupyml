# Scoreboard — hyperbolic_hierarchy

**Goal:** Embed a tree and predict held-out pairwise graph distances; Spearman rho.

**Metric:** `spearman_rho` (higher is better) · **QA floor:** `0.55`

| Rank | Submission | Score | Runtime (s) |
|-----:|------------|------:|------------:|
| 1 | `baseline_graph_distance` | 1.0000 | 0.11 |
| 2 | `baseline_lorentz` | 0.7842 | 5.07 |
| 3 | `baseline_poincare` | 0.7755 | 3.32 |

_Regenerate with `python bench/harness.py hyperbolic_hierarchy`._
