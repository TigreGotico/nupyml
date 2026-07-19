"""A glass-box GA2M: boosted per-feature shape functions (Lou et al., 2013)."""
import numpy as np
from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone)
from ..utils import check_array, check_random_state
from ..tree import DecisionTreeRegressor, DecisionTreeClassifier


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class ExplainableBoostingClassifier(BaseEstimator, ClassifierMixin):
    """A glass-box GA2M: boosted per-feature shape functions (Lou et al., 2013).

    A full gradient-boosted forest is accurate but opaque. An EBM keeps the
    boosting but restricts each weak learner to a SINGLE feature, cycling through
    features round-robin. The model is then a sum of one-dimensional SHAPE
    FUNCTIONS -- ``f(x) = b + sum_j f_j(x_j)`` -- so you can plot exactly how each
    feature moves the prediction, an additive model with the accuracy of boosting.
    Binary classification via logistic loss; ``feature_shapes_`` exposes each
    learned ``f_j`` as a small regression tree ensemble.
    """

    def __init__(self, n_rounds=200, learning_rate=0.1, max_leaf_nodes=8,
                 random_state=None):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        yb = (y == self.classes_[1]).astype(float)
        n, d = X.shape
        self.intercept_ = np.log((yb.mean() + 1e-6) / (1 - yb.mean() + 1e-6))
        F = np.full(n, self.intercept_)
        self.shapes_ = [[] for _ in range(d)]           # per-feature tree list
        rng = check_random_state(self.random_state)
        for r in range(self.n_rounds):
            j = r % d                                    # round-robin over features
            p = _sigmoid(F)
            grad = p - yb                                # logistic gradient
            tree = DecisionTreeRegressor(max_depth=3, random_state=rng)
            tree.fit(X[:, [j]], -grad)
            update = self.learning_rate * tree.predict(X[:, [j]])
            F += update
            self.shapes_[j].append(tree)
        return self

    def _score(self, X):
        F = np.full(len(X), self.intercept_)
        for j, trees in enumerate(self.shapes_):
            for tree in trees:
                F += self.learning_rate * tree.predict(X[:, [j]])
        return F

    def shape_function(self, j, grid):
        """The learned contribution of feature ``j`` over a 1-D grid of values."""
        grid = np.asarray(grid).reshape(-1, 1)
        out = np.zeros(len(grid))
        for tree in self.shapes_[j]:
            out += self.learning_rate * tree.predict(grid)
        return out

    def predict_proba(self, X):
        X = check_array(X)
        p = _sigmoid(self._score(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self._score(check_array(X)) > 0).astype(int)]


__all__ = ["ExplainableBoostingClassifier"]
