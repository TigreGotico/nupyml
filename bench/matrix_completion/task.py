"""Matrix completion: recover a low-rank matrix from a few observed entries.

The observed cells are encoded as ``X`` rows of ``[row, col]`` with the value in
``y``; the test set is held-out cells. A low-rank structure means a factorization
or nuclear-norm method should reconstruct the missing entries far better than the
global mean. Scored by RMSE (lower is better).
"""
import numpy as np

from nupyml.metrics import mean_squared_error

KIND = "supervised"
GOAL = "Fill held-out cells of a low-rank matrix from observed ones; RMSE."
METRIC = "rmse"
HIGHER_IS_BETTER = False
MIN_SCORE = 1.0           # ceiling: an entry must reach RMSE <= this to pass


def load():
    rng = np.random.RandomState(0)
    n_rows, n_cols, rank = 40, 30, 3
    M = rng.normal(0, 1, (n_rows, rank)) @ rng.normal(0, 1, (rank, n_cols))
    cells = [(i, j) for i in range(n_rows) for j in range(n_cols)
             if rng.rand() < 0.45]
    rng.shuffle(cells)
    n_tr = int(0.8 * len(cells))
    tr, te = cells[:n_tr], cells[n_tr:]
    noise = 0.05
    X_train = np.array(tr)
    y_train = np.array([M[i, j] + noise * rng.randn() for i, j in tr])
    X_test = np.array(te)
    y_test = np.array([M[i, j] for i, j in te])
    return X_train, y_train, X_test, y_test


def metric(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
