"""Quantile regression forest: the WHOLE conditional distribution, not the mean.

THE INSIGHT (Meinshausen, 2006)
-------------------------------
A random forest regressor averages, per leaf, the training targets that fall in
it -- and then averages across trees -- so it reports only the conditional MEAN.
But each leaf still HOLDS all of its training targets; a quantile forest simply
keeps them instead of collapsing to their mean. To predict a quantile at a new
point, gather the training targets from the leaves the point lands in across all
trees, and read off the empirical quantile of that pooled set.

WHY IT MATTERS
--------------
The mean tells you nothing about spread. A quantile forest gives calibrated
PREDICTION INTERVALS ([q_0.05, q_0.95]) and reveals heteroscedasticity -- where
the model is confident and where it is not -- from the same trained forest, no
distributional assumption required.

IMPLEMENTATION NOTE
-------------------
Leaves are identified by their prediction VALUE: all samples in a leaf get that
leaf's mean, so training samples sharing a tree's prediction share its leaf. (Two
distinct leaves with a byte-identical mean would merge -- vanishingly rare with
real-valued targets, and only mildly blurs the pooled set if it happens.)
"""
import numpy as np

from ..base import BaseEstimator, RegressorMixin, check_is_fitted
from ..utils import check_X_y, check_array


class QuantileForest(BaseEstimator, RegressorMixin):
    """Random forest that answers ``predict_quantile`` for any quantile."""

    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=5,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        from ..tree import DecisionTreeRegressor
        from ..utils import check_random_state
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        n = len(y)
        self.trees_ = []
        self.leaf_targets_ = []          # per tree: {leaf prediction value -> y array}
        for _ in range(self.n_estimators):
            idx = rng.choice(n, n, replace=True)             # bootstrap
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features="sqrt", random_state=rng.randint(2 ** 31 - 1))
            tree.fit(X[idx], y[idx])
            leaf_pred = np.round(tree.predict(X[idx]), 8)    # leaf id by value
            mapping = {}
            for v, target in zip(leaf_pred, y[idx]):
                mapping.setdefault(v, []).append(target)
            self.trees_.append(tree)
            self.leaf_targets_.append({v: np.array(t) for v, t in mapping.items()})
        return self

    def _pooled(self, x_row):
        """All training targets from the leaves x lands in, across the forest."""
        pooled = []
        xr = x_row.reshape(1, -1)
        for tree, mapping in zip(self.trees_, self.leaf_targets_):
            v = round(float(tree.predict(xr)[0]), 8)
            arr = mapping.get(v)
            if arr is not None:
                pooled.append(arr)
        return np.concatenate(pooled) if pooled else np.array([0.0])

    def predict_quantile(self, X, quantile=0.5):
        """Conditional quantile(s) at each row. ``quantile`` may be a scalar or a
        list; returns shape ``(n,)`` or ``(n, len(quantile))``."""
        check_is_fitted(self, "trees_")
        X = check_array(X)
        qs = np.atleast_1d(quantile)
        out = np.empty((len(X), len(qs)))
        for i, row in enumerate(X):
            pooled = self._pooled(row)
            out[i] = np.quantile(pooled, qs)
        return out[:, 0] if np.isscalar(quantile) else out

    def predict(self, X):
        """The conditional median (a robust point prediction)."""
        return self.predict_quantile(X, 0.5)

    def predict_interval(self, X, coverage=0.9):
        alpha = 1 - coverage
        both = self.predict_quantile(X, [alpha / 2, 1 - alpha / 2])
        return both[:, 0], both[:, 1]


__all__ = ["QuantileForest"]
