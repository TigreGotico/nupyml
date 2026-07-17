# fraud_imbalanced

**Goal:** detect a rare, overlapping positive class (~5% of the data).

- **Kind:** supervised — `solve(X_train, y_train, X_test) -> y_pred`
- **Data:** `make_classification` with low class separation, positives subsampled to ~5%, stratified 70/30 split.
- **Metric:** **macro-F1** (higher is better) — deliberately *not* accuracy.
- **QA floor:** 0.55. A strong entry reaches ~0.77.

Accuracy is useless here: predicting "not fraud" for everyone scores ~95% and catches nothing. Macro-F1 averages the two classes' F1, so ignoring the rare class is punished. Resampling (SMOTE) is the textbook remedy but, on this hard overlapping data, barely helps or even hurts — the scoreboard is the honest judge, and gradient boosting currently leads.
