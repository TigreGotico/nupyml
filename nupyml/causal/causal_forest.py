"""How the treatment effect VARIES across people (Wager & Athey, 2018)."""
import numpy as np
from ..base import BaseEstimator, clone
from ..utils import check_array, check_random_state


class CausalForest(BaseEstimator):
    """How the treatment effect VARIES across people (Wager & Athey, 2018).

    An average treatment effect hides that a drug may help some patients and harm
    others. A causal forest estimates the CONDITIONAL effect ``tau(x)`` by growing
    many trees that split to separate regions of DIFFERENT effect (not different
    outcome), then, in each leaf, estimating the effect as treated-minus-control
    mean. "Honesty" -- using separate samples to choose splits and to estimate
    effects -- keeps the estimates unbiased. Averaging over trees gives a smooth,
    individualised ``tau(x)`` with far lower variance than a single tree. Assumes
    (approximately) randomised treatment.
    """

    def __init__(self, n_estimators=100, max_depth=4, min_leaf=5,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.random_state = random_state

    def _grow(self, X, t, y, depth, rng):
        node = {"tau": self._leaf_effect(t, y)}
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf:
            return node
        best = None
        feats = rng.choice(X.shape[1], max(1, int(np.sqrt(X.shape[1]))),
                           replace=False)
        for j in feats:
            for thr in np.quantile(X[:, j], [0.3, 0.5, 0.7]):
                left = X[:, j] <= thr
                if (left.sum() < self.min_leaf or (~left).sum() < self.min_leaf
                        or t[left].std() == 0 or t[~left].std() == 0):
                    continue
                tl = self._leaf_effect(t[left], y[left])
                tr = self._leaf_effect(t[~left], y[~left])
                het = left.sum() * (tl) ** 2 + (~left).sum() * (tr) ** 2  # heterogeneity
                if best is None or het > best[0]:
                    best = (het, j, thr, left)
        if best is None:
            return node
        _, j, thr, left = best
        node.update({"dim": j, "thr": thr,
                     "left": self._grow(X[left], t[left], y[left], depth + 1, rng),
                     "right": self._grow(X[~left], t[~left], y[~left], depth + 1, rng)})
        return node

    @staticmethod
    def _leaf_effect(t, y):
        if (t == 1).any() and (t == 0).any():
            return y[t == 1].mean() - y[t == 0].mean()
        return 0.0

    def fit(self, X, treatment, y):
        X = check_array(X)
        t = np.asarray(treatment); y = np.asarray(y, float)
        rng = check_random_state(self.random_state)
        n = len(y)
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, n, replace=True)
            self.trees_.append(self._grow(X[idx], t[idx], y[idx], 0, rng))
        return self

    def _predict_tree(self, node, x):
        while "dim" in node:
            node = node["left"] if x[node["dim"]] <= node["thr"] else node["right"]
        return node["tau"]

    def predict(self, X):
        X = check_array(X)
        return np.array([np.mean([self._predict_tree(t, x) for t in self.trees_])
                         for x in X])


__all__ = ["CausalForest"]
