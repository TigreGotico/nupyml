# gp_regression

**Goal:** regress a smooth signal with a scalable Gaussian process.

- **Kind:** supervised — `solve(X_train, y_train, X_test) -> y_pred`
- **Data:** 600 points of `sin(x) + 0.3 sin(3x)` on `[-4, 4]` with light noise; 30% held out.
- **Metric:** R² (higher is better).
- **QA floor:** 0.85. A sparse (inducing-point) GP matches the exact GP at a fraction of the cost; both, and even kNN, clear ~0.99 here.

The point of interest is scalability: exact GP inference is O(n³), so `SparseVariationalGP` should reproduce exact quality with a handful of inducing points.
