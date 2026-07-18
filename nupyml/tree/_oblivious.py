"""Oblivious (symmetric) decision trees and trees with linear leaves.

The base ``tree`` grows ordinary CART trees where every node picks its own split.
These are two structured alternatives: the oblivious tree (one split per LEVEL,
shared by all nodes) and the linear-leaf tree (constant leaves replaced by linear
models).
"""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted
from ..utils import check_X_y, check_array
from ..preprocessing import LabelEncoder


class _ObliviousBase(BaseEstimator):
    """An oblivious/symmetric tree: at each depth, ONE (feature, threshold) is used
    by every node. A sample's leaf is the bitstring of its ``depth`` comparisons.

    WHY SYMMETRIC (CatBoost's default)
    ----------------------------------
    A normal tree chooses a different split at every node, which is expressive but
    high-variance and slow to evaluate. An oblivious tree forces the SAME split
    across a whole level, so the whole tree is just ``depth`` comparisons and the
    leaf index is a ``depth``-bit number -- evaluation is a handful of operations
    with no branching, and the heavy constraint acts as strong regularisation
    (the reason CatBoost uses them by default and resists overfitting).
    """

    def __init__(self, max_depth=4, n_thresholds=32):
        self.max_depth = max_depth
        self.n_thresholds = n_thresholds

    def _fit_tree(self, X, y, impurity, leaf_value):
        n, d = X.shape
        self.splits_ = []                             # (feature, threshold) per level
        leaf_idx = np.zeros(n, dtype=int)             # current leaf id per sample
        for level in range(self.max_depth):
            best = (np.inf, None, None)
            n_leaves = 2 ** level
            for f in range(d):
                qs = np.quantile(X[:, f], np.linspace(0.05, 0.95, self.n_thresholds))
                for thr in np.unique(qs):
                    # applying this ONE split to every current leaf, total impurity
                    new_idx = leaf_idx * 2 + (X[:, f] > thr)
                    total = 0.0
                    for g in range(2 * n_leaves):
                        m = new_idx == g
                        if m.any():
                            total += m.sum() * impurity(y[m])
                    if total < best[0]:
                        best = (total, f, thr)
            _, f, thr = best
            self.splits_.append((f, thr))
            leaf_idx = leaf_idx * 2 + (X[:, f] > thr)
        # leaf values from the final assignment
        self.leaf_values_ = {}
        for g in np.unique(leaf_idx):
            self.leaf_values_[int(g)] = leaf_value(y[leaf_idx == g])
        self._default = leaf_value(y)
        return self

    def _leaf_index(self, X):
        idx = np.zeros(len(X), dtype=int)
        for f, thr in self.splits_:
            idx = idx * 2 + (X[:, f] > thr)
        return idx


class ObliviousDecisionTreeClassifier(_ObliviousBase, ClassifierMixin):
    """Oblivious decision tree for classification (majority-vote leaves)."""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        yi = self._le.transform(y)

        def gini(labels):
            _, c = np.unique(labels, return_counts=True)
            p = c / c.sum()
            return 1 - np.sum(p ** 2)

        def majority(labels):
            return int(np.bincount(labels, minlength=len(self.classes_)).argmax())

        self._fit_tree(X, yi, gini, majority)
        return self

    def predict(self, X):
        check_is_fitted(self, "splits_")
        X = check_array(X)
        idx = self._leaf_index(X)
        out = np.array([self.leaf_values_.get(int(g), self._default) for g in idx])
        return self.classes_[out]


class ObliviousDecisionTreeRegressor(_ObliviousBase, RegressorMixin):
    """Oblivious decision tree for regression (mean-value leaves)."""

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self._fit_tree(X, y, lambda v: np.var(v) if len(v) else 0.0,
                       lambda v: float(np.mean(v)) if len(v) else 0.0)
        return self

    def predict(self, X):
        check_is_fitted(self, "splits_")
        idx = self._leaf_index(check_array(X))
        return np.array([self.leaf_values_.get(int(g), self._default) for g in idx])


class LinearTreeRegressor(BaseEstimator, RegressorMixin):
    """A regression tree with LINEAR models in the leaves (piecewise-linear).

    A CART regressor predicts a CONSTANT per leaf, so approximating a smooth trend
    needs many tiny leaves (a staircase). A linear-tree fits a LINEAR model in each
    leaf instead, so a few leaves capture a piecewise-linear surface -- far fewer
    splits for the same fit, and predictions that extrapolate within a leaf rather
    than flattening. Here a shallow tree partitions the space and a ridge
    regression is fit per leaf.
    """

    def __init__(self, max_depth=3, min_samples_leaf=10, alpha=1.0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha

    def _split(self, X, y, depth):
        from ..linear_model import Ridge
        node = {}
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            node["leaf"] = Ridge(alpha=self.alpha).fit(X, y)
            return node
        best = (np.inf, None, None)
        for f in range(X.shape[1]):
            for thr in np.quantile(X[:, f], np.linspace(0.1, 0.9, 9)):
                left = X[:, f] <= thr
                if left.sum() < self.min_samples_leaf or (~left).sum() < self.min_samples_leaf:
                    continue
                # variance-reduction score for this split
                err = left.sum() * np.var(y[left]) + (~left).sum() * np.var(y[~left])
                if err < best[0]:
                    best = (err, f, thr)
        if best[1] is None:
            from ..linear_model import Ridge
            node["leaf"] = Ridge(alpha=self.alpha).fit(X, y)
            return node
        _, f, thr = best
        node.update(feature=f, threshold=thr)
        left = X[:, f] <= thr
        node["left"] = self._split(X[left], y[left], depth + 1)
        node["right"] = self._split(X[~left], y[~left], depth + 1)
        return node

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.tree_ = self._split(X, y, 0)
        return self

    def _predict_one(self, node, x):
        while "leaf" not in node:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return float(node["leaf"].predict(x[None])[0])

    def predict(self, X):
        check_is_fitted(self, "tree_")
        X = check_array(X)
        return np.array([self._predict_one(self.tree_, x) for x in X])


__all__ = ["ObliviousDecisionTreeClassifier", "ObliviousDecisionTreeRegressor",
           "LinearTreeRegressor"]
