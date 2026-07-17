"""Recommender rating prediction: fill in held-out (user, item) ratings.

The training data is a set of observed ratings, encoded as ``X_train`` rows of
``[user_id, item_id]`` with the rating in ``y_train``. The test set is another
set of (user, item) pairs whose ratings are hidden; predict them. The underlying
matrix is low-rank, so a latent-factor model should clearly beat predicting the
global mean. Scored by RMSE -- LOWER is better.
"""
import numpy as np

from nupyml.metrics import mean_squared_error

KIND = "supervised"
GOAL = ("Predict held-out (user, item) ratings from a low-rank matrix; "
        "scored by RMSE (lower is better).")
METRIC = "rmse"
HIGHER_IS_BETTER = False
MIN_SCORE = 0.85          # ceiling: an entry must get RMSE <= this to pass QA


def load():
    rng = np.random.RandomState(0)
    n_users, n_items, k = 80, 50, 4
    P = rng.normal(0, 1, size=(n_users, k))
    Q = rng.normal(0, 1, size=(n_items, k))
    full = P @ Q.T                                   # the true low-rank ratings
    # observe ~35% of cells; split those into train / test
    pairs = [(u, i) for u in range(n_users) for i in range(n_items)
             if rng.rand() < 0.35]
    rng.shuffle(pairs)
    n_tr = int(0.8 * len(pairs))
    tr, te = pairs[:n_tr], pairs[n_tr:]
    noise = 0.1
    X_train = np.array(tr)
    y_train = np.array([full[u, i] + noise * rng.randn() for u, i in tr])
    X_test = np.array(te)
    y_test = np.array([full[u, i] for u, i in te])
    return X_train, y_train, X_test, y_test


def metric(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
