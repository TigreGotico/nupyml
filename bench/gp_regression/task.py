"""GP regression: fit a smooth function with calibrated uncertainty.

A held-out regression on a smooth signal. The point of interest is scalability: an
exact Gaussian process is O(n^3), so the task is sized where a sparse (inducing-point)
GP should match exact quality at a fraction of the cost. Scored by R^2.
"""
import numpy as np

from nupyml.metrics import r2_score
from nupyml.model_selection import train_test_split

KIND = "supervised"
GOAL = "Regress a smooth multi-frequency signal; scored by held-out R^2."
METRIC = "r2"
HIGHER_IS_BETTER = True
MIN_SCORE = 0.85


def load():
    rng = np.random.RandomState(0)
    X = np.sort(rng.uniform(-4, 4, (600, 1)), axis=0)
    y = (np.sin(X[:, 0]) + 0.3 * np.sin(3 * X[:, 0])) + 0.05 * rng.randn(600)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    return X_train, y_train, X_test, y_test


def metric(y_true, y_pred):
    return r2_score(y_true, y_pred)
