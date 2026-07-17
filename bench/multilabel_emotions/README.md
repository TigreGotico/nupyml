# multilabel_emotions

**Goal.** Predict a *set* of labels per sample, where the labels are correlated.

**Data.** 600 synthetic samples, 20 features, 5 labels. Each label is a logistic
function of a random projection of the features *plus* shared latent factors that
make labels co-occur — so a method that models label correlation has something to
gain. Deterministic 70/30 split (seed 0).

**KIND.** `supervised` — `solve(X_train, y_train, X_test) -> y_pred`, where
`y_train` and the returned `y_pred` are `(n_samples, n_labels)` 0/1 matrices.

**Metric.** Macro-averaged F1 across the label columns (higher is better). QA
floor: **0.55**.

**Baselines.**
- `baseline_binary_relevance` — one independent classifier per label (the
  correlation-blind floor).
- `baseline_classifier_chain` — chains the classifiers so each sees earlier
  labels.
- `baseline_mlknn` — Bayesian k-NN over the label neighbourhood.

Add a submission by dropping `<name>.py` with a `solve(X_train, y_train, X_test)`
returning a 0/1 label matrix into `submissions/`.
