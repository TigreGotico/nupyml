"""Isolation forest with OBLIQUE cuts instead of axis-parallel (Hariri, 2019)."""
import numpy as np
from ..utils import check_array, check_random_state
from .detectors import _Detector


class ExtendedIsolationForest(_Detector):
    """Isolation forest with OBLIQUE cuts instead of axis-parallel (Hariri, 2019).

    A standard isolation forest cuts one feature at a time, so its decision regions
    are boxes -- and on data that is not axis-aligned this leaves ghostly bands of
    artificially low/high anomaly score parallel to the axes. The extended version
    cuts with a RANDOM HYPERPLANE (a random normal direction and a random
    intercept) at each node, removing the bias. Everything else is identical: an
    anomaly is isolated by fewer cuts, so a SHORTER average path length means a
    higher anomaly score.
    """

    def __init__(self, n_estimators=100, max_samples=256, contamination=0.1,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

    def _build(self, X, depth, max_depth, rng):
        n = len(X)
        if depth >= max_depth or n <= 1:
            return {"size": n}
        normal = rng.randn(X.shape[1])                 # random hyperplane direction
        proj = X @ normal
        p = rng.uniform(proj.min(), proj.max())        # random intercept
        left = proj < p
        if left.all() or (~left).all():
            return {"size": n}
        return {"normal": normal, "p": p,
                "left": self._build(X[left], depth + 1, max_depth, rng),
                "right": self._build(X[~left], depth + 1, max_depth, rng)}

    @staticmethod
    def _c(n):
        if n <= 1:
            return 0.0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n

    def _path(self, x, node, depth):
        if "normal" not in node:
            return depth + self._c(node["size"])
        branch = node["left"] if x @ node["normal"] < node["p"] else node["right"]
        return self._path(x, branch, depth + 1)

    def fit(self, X):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        m = min(self.max_samples, len(X))
        max_depth = int(np.ceil(np.log2(max(m, 2))))
        self.trees_ = []
        for _ in range(self.n_estimators):
            sub = X[rng.choice(len(X), m, replace=False)]
            self.trees_.append(self._build(sub, 0, max_depth, rng))
        self._norm = self._c(m)
        self._set_threshold(self.decision_function(X))
        return self

    def decision_function(self, X):
        X = check_array(X)
        paths = np.array([[self._path(x, t, 0) for t in self.trees_] for x in X])
        avg = paths.mean(axis=1)
        return 2.0 ** (-avg / self._norm)              # short path -> high score


__all__ = ["ExtendedIsolationForest"]
