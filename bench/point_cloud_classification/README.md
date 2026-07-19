# point_cloud_classification

**Goal:** classify an unordered 3-D point cloud by its shape (sphere vs cube surface).

- **Kind:** supervised — `solve(X_train, y_train, X_test) -> y_pred`, with `X` of shape `(n, n_points, 3)`.
- **Data:** 400 clouds of 32 points each, randomly rotated; 30% held out.
- **Metric:** accuracy (higher is better).
- **QA floor:** 0.80.

The classifier must be invariant to point **order**. `PointNet` (shared per-point MLP + symmetric max-pool) learns this directly; a hand-crafted bag-of-points baseline reaches it through radial statistics.
