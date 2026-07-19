"""Half-space trees: isolation by random axis-aligned splits, scored by MASS."""
import numpy as np
from ..utils import check_array, check_random_state
from .detectors import _Detector


class HalfSpaceTrees(_Detector):
    """Half-space trees: isolation by random axis-aligned splits, scored by MASS.

    A half-space tree splits space by picking a random feature and a random
    threshold within its range, recursively, to a fixed depth -- with NO reference
    to the data (the structure is random). Each training point falls into a leaf;
    a leaf's MASS is how many training points share it. A test point landing in a
    low-mass leaf sits where little training data lived -- an anomaly. Averaging
    the (depth-weighted) mass over many random trees gives the score.

    Like isolation forest it needs no distances and scales linearly, and its
    data-independent structure is what makes the streaming version (updating leaf
    masses as data flows) trivial.

    Tan, Ting & Liu (2011).
    """

    def __init__(self, n_estimators=25, max_depth=8, contamination=0.1,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.contamination = contamination
        self.random_state = random_state

    def _build(self, lo, hi, depth, rng):
        if depth >= self.max_depth:
            return {"leaf": True, "mass": 0}
        j = rng.randint(len(lo))
        split = rng.uniform(lo[j], hi[j])
        left_hi = hi.copy(); left_hi[j] = split
        right_lo = lo.copy(); right_lo[j] = split
        return {"leaf": False, "feature": j, "split": split,
                "left": self._build(lo, left_hi, depth + 1, rng),
                "right": self._build(right_lo, hi, depth + 1, rng)}

    def _fill(self, node, X):
        if node["leaf"]:
            node["mass"] = len(X)
            return
        left = X[:, node["feature"]] <= node["split"]
        self._fill(node["left"], X[left])
        self._fill(node["right"], X[~left])

    def _leaf_mass(self, node, x):
        if node["leaf"]:
            return node["mass"]
        side = "left" if x[node["feature"]] <= node["split"] else "right"
        return self._leaf_mass(node[side], x)

    def fit(self, X):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        # randomise the workspace a little beyond the data range (HST convention)
        span = X.max(axis=0) - X.min(axis=0) + 1e-9
        lo = X.min(axis=0) - 0.2 * span
        hi = X.max(axis=0) + 0.2 * span
        self.trees_ = []
        for _ in range(self.n_estimators):
            tree = self._build(lo, hi, 0, rng)
            self._fill(tree, X)
            self.trees_.append(tree)
        self._n_train = len(X)
        self._set_threshold(self.decision_function(X))
        return self

    def decision_function(self, X):
        X = check_array(X)
        scores = np.empty(len(X))
        for i, x in enumerate(X):
            mass = np.mean([self._leaf_mass(t, x) for t in self.trees_])
            scores[i] = -mass                        # low mass -> anomalous
        return scores


__all__ = ["HalfSpaceTrees"]
