# anomaly_detection

**Goal.** Rank points by how anomalous they are, with no labels.

**Data.** 1000 points in 8-D: 950 inliers drawn from two tight Gaussian clusters,
plus 50 outliers spread uniformly across a much wider box so they land in the
sparse space around and between the clusters. The outlier labels are held out for
scoring only. Deterministic (seed 0).

**KIND.** `clustering` — `solve(X) -> scores`, one real-valued anomaly score per
point (**higher = more anomalous**). No threshold needed.

**Metric.** ROC-AUC of the scores against the hidden outlier labels (higher is
better). QA floor: **0.85**.

**Baselines.**
- `baseline_hbos` — histogram-based, assumes feature independence.
- `baseline_ecod` — empirical-CDF tail probabilities, parameter-free.
- `baseline_knn` — distance to the k-th nearest neighbour.
- `baseline_isolation_forest` — average isolation path length.

Add a submission by dropping `<name>.py` with a `solve(X)` returning a score per
row into `submissions/`.
