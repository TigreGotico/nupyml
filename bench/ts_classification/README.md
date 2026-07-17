# ts_classification

**Goal.** Classify each 1-D time series by the pattern that generated it.

**Data.** 360 synthetic series of length 64, three classes: a 3-cycle sine, a
6-cycle sine, and a single Gaussian bump — each with a **random phase / bump
location** and additive noise. Deterministic 70/30 train/test split (seed 0). The
randomised phase is what defeats a plain tabular model and rewards a
time-series-aware one.

**KIND.** `supervised` — `solve(X_train, y_train, X_test) -> y_pred`, where each
row of `X` is one series.

**Metric.** Accuracy (higher is better). QA floor: **0.80**.

**Baselines.**
- `baseline_dtw_knn` — 1-NN under banded dynamic time warping.
- `baseline_features_rf` — phase-invariant summary features + random forest.
- `baseline_flatten_svc` — series as a flat vector, StandardScaler + RBF SVC
  (the location-sensitive floor).

Add a submission by dropping `<name>.py` with a `solve(X_train, y_train, X_test)`
into `submissions/` and running `python bench/harness.py ts_classification`.
