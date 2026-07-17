# moons_classification

**Goal:** separate two interleaving half-moons — a nonlinearly-separable binary problem.

- **Kind:** supervised — `solve(X_train, y_train, X_test) -> y_pred`
- **Data:** `make_moons(n_samples=1000, noise=0.2, random_state=0)`, stratified 70/30 split.
- **Metric:** accuracy (higher is better).
- **QA floor:** 0.90. A strong entry reaches ~0.97.

No straight line separates the classes, so a linear model tops out near 0.88 while kernel SVMs, trees, kNN and MLPs reach ~0.97. The task rewards a model that can bend.
