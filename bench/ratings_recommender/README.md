# ratings_recommender

**Goal.** Predict held-out `(user, item)` ratings from a sparsely observed,
low-rank rating matrix.

**Data.** 80 users × 50 items, true ratings generated from rank-4 latent factors
plus small noise. About 35% of cells are observed and split 80/20 into
train/test. Encoded for the harness as `X` rows of `[user_id, item_id]` with the
rating in `y`. Deterministic (seed 0).

**KIND.** `supervised` — `solve(X_train, y_train, X_test) -> y_pred`. Build the
model from the training pairs/ratings, then predict a rating for each test pair.

**Metric.** RMSE — **lower is better**. QA ceiling: an entry must reach RMSE
**≤ 0.85** to pass.

**Baselines.**
- `baseline_global_mean` — predict the mean rating for everything (the floor).
- `baseline_matrix_factorization` — SGD latent factors + biases.
- `baseline_svdpp` — factorization that also uses which items a user rated.
- `baseline_item_cf` — item-based neighbourhood collaborative filtering.

Add a submission by dropping `<name>.py` with a `solve(X_train, y_train, X_test)`
into `submissions/`. A robust entry should fall back gracefully for users or
items unseen in training.
