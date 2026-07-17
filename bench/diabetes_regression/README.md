# diabetes_regression

**Goal:** predict quantitative diabetes progression one year on from ten baseline measurements.

- **Kind:** supervised — `solve(X_train, y_train, X_test) -> y_pred`
- **Data:** the bundled `load_diabetes` set (442 samples, 10 features), 70/30 split at `random_state=0`.
- **Metric:** R² (higher is better).
- **QA floor:** 0.15 (a sanity floor — this dataset is genuinely hard). A strong entry reaches ~0.45.

A famously low-ceiling regression problem: the signal is weak and the features are few, so ridge regression is a strong baseline and elaborate models rarely help. RBF kernel ridge needs feature scaling to be competitive — a lesson the baseline leaves on the table.
