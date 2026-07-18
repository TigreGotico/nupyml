"""Multi-output regression: predict a VECTOR target with correlated components.

The three targets share the same informative features (and some shared latent
structure), so a joint model can borrow strength across outputs. ``y_train`` and
the returned ``y_pred`` are ``(n_samples, n_outputs)``. Scored by the average R²
across the outputs.
"""
import numpy as np

from nupyml.metrics import r2_score

KIND = "supervised"
GOAL = "Predict a 3-dimensional target; scored by average R^2 over the outputs."
METRIC = "avg_r2"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.65


def load():
    rng = np.random.RandomState(0)
    n, d, k = 600, 8, 3
    X = rng.randn(n, d)
    W = rng.randn(d, k)
    shared = rng.randn(n, 2) @ rng.randn(2, k)        # shared latent structure
    Y = X @ W + 0.5 * shared + 0.3 * rng.randn(n, k)
    perm = rng.permutation(n)
    X, Y = X[perm], Y[perm]
    n_tr = int(0.7 * n)
    return X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:]


def metric(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean([r2_score(y_true[:, k], y_pred[:, k])
                          for k in range(y_true.shape[1])]))
