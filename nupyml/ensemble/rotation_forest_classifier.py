"""Decorrelate trees by ROTATING each one's feature space (Rodriguez, 2006)."""
import numpy as np
from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone)
from ..utils import check_array, check_random_state
from ..tree import DecisionTreeRegressor, DecisionTreeClassifier
from ..decomposition import PCA


class RotationForestClassifier(BaseEstimator, ClassifierMixin):
    """Decorrelate trees by ROTATING each one's feature space (Rodriguez, 2006).

    Random forests decorrelate trees by sampling features; Rotation Forest does it
    by ROTATION. For each tree it splits the features into random subsets, runs PCA
    on a bootstrap of each subset, and trains the tree on the PCA-ROTATED features.
    Because axis-parallel splits in a rotated space are oblique in the original, the
    trees are both diverse AND individually strong -- the combination that often
    edges out plain forests on structured data.
    """

    def __init__(self, n_estimators=25, n_subsets=3, max_depth=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.n_subsets = n_subsets
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_rotation(self, X, rng):
        d = X.shape[1]
        perm = rng.permutation(d)
        subsets = np.array_split(perm, self.n_subsets)
        R = np.zeros((d, d))
        for sub in subsets:
            if len(sub) == 0:
                continue
            boot = X[rng.choice(len(X), len(X), replace=True)][:, sub]
            pca = PCA().fit(boot)
            comp = pca.components_.T                     # (len(sub), len(sub))
            for a, ia in enumerate(sub):
                for b, ib in enumerate(sub):
                    R[ia, ib] = comp[a, b]
        return R

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = check_random_state(self.random_state)
        self.rotations_, self.trees_ = [], []
        for _ in range(self.n_estimators):
            R = self._make_rotation(X, rng)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=rng)
            tree.fit(X @ R, y)
            self.rotations_.append(R); self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        X = check_array(X)
        votes = np.zeros((len(X), len(self.classes_)))
        for R, tree in zip(self.rotations_, self.trees_):
            pred = tree.predict(X @ R)
            for i, c in enumerate(self.classes_):
                votes[:, i] += (pred == c)
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


__all__ = ["RotationForestClassifier"]
