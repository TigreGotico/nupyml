# link_prediction

**Kind:** supervised · **Metric:** ROC-AUC (higher is better) · **Floor:** 0.75

Predict which node pairs are linked. A community graph is built and some edges
hidden; each pair is described by classic link-prediction scores (common neighbours,
Jaccard, Adamic-Adar, preferential attachment) computed on the observed graph, and
the task is to rank true (hidden) edges above non-edges.

`solve(X_train, y_train, X_test) -> link_scores`

Baselines: `logistic`, `random_forest`, `adamic_adar` (single-feature).
