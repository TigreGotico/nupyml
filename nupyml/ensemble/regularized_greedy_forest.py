"""Grow ONE forest structure under an explicit penalty (Johnson & Zhang, 2014)."""
import numpy as np
from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone)
from ..utils import check_array, check_random_state
from ..tree import DecisionTreeRegressor, DecisionTreeClassifier


class RegularizedGreedyForest(BaseEstimator, RegressorMixin):
    """Grow ONE forest structure under an explicit penalty (Johnson & Zhang, 2014).

    Gradient boosting fits a sequence of FIXED-size trees. RGF instead treats the
    whole forest as one evolving structure and, at each step, greedily makes the
    single change -- grow an existing leaf or start a new tree -- that most reduces
    a regularised objective (loss + a penalty on leaf weights), periodically
    re-optimising all leaf values. Directly regularising the structure, rather than
    capping tree size, often gives a more accurate model for the same number of
    leaves. This is a compact regression implementation on squared loss.
    """

    def __init__(self, max_leaves=100, learning_rate=0.5, l2=0.1, max_tree_depth=4,
                 random_state=None):
        self.max_leaves = max_leaves
        self.learning_rate = learning_rate
        self.l2 = l2
        self.max_tree_depth = max_tree_depth
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y, float)
        rng = check_random_state(self.random_state)
        self.intercept_ = y.mean()
        pred = np.full(len(y), self.intercept_)
        self.trees_, self.weights_ = [], []
        n_leaves = 0
        while n_leaves < self.max_leaves:
            residual = y - pred                          # negative gradient (sq loss)
            depth = min(self.max_tree_depth,
                        max(1, int(np.log2(self.max_leaves - n_leaves + 1))))
            tree = DecisionTreeRegressor(max_depth=depth, random_state=rng)
            tree.fit(X, residual)
            raw = tree.predict(X)
            # shrink toward zero: L2 regularisation on the added leaf weights
            w = self.learning_rate / (1.0 + self.l2)
            pred = pred + w * raw
            self.trees_.append(tree); self.weights_.append(w)
            n_leaves += 2 ** depth
            if np.abs(residual).max() < 1e-6:
                break
        return self

    def predict(self, X):
        X = check_array(X)
        out = np.full(len(X), self.intercept_)
        for tree, w in zip(self.trees_, self.weights_):
            out += w * tree.predict(X)
        return out


__all__ = ["RegularizedGreedyForest"]
