"""Decorrelate members by giving each a FEATURE subset (Ho, 1998)."""
import numpy as np
from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone)
from ..utils import check_array, check_random_state
from ..tree import DecisionTreeClassifier


class RandomSubspaceClassifier(BaseEstimator, ClassifierMixin):
    """Decorrelate members by giving each a FEATURE subset (Ho, 1998).

    Bagging resamples rows; the random subspace method resamples COLUMNS. Each member
    is trained on a random subset of the features, so different members key on
    different signals and their errors decorrelate -- exactly the diversity that
    makes averaging work. It shines when there are many features, some redundant
    (text, genomics, images), and is the "feature-bagging" half of what a random
    forest does. Any base estimator; predictions are averaged over members, each
    seeing only its own feature subset.
    """

    def __init__(self, estimator=None, n_estimators=25, max_features=0.5,
                 random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y)
        self.classes_ = np.unique(y)
        rng = check_random_state(self.random_state)
        d = X.shape[1]
        k = max(1, int(self.max_features * d) if isinstance(self.max_features, float)
                else self.max_features)
        base = self.estimator if self.estimator is not None else DecisionTreeClassifier()
        self.subsets_, self.models_ = [], []
        for _ in range(self.n_estimators):
            cols = rng.choice(d, k, replace=False)       # a random feature subspace
            m = clone(base)
            if "random_state" in m.get_params():
                m.set_params(random_state=rng.randint(1 << 30))
            m.fit(X[:, cols], y)
            self.subsets_.append(cols); self.models_.append(m)
        return self

    def predict_proba(self, X):
        X = check_array(X)
        votes = np.zeros((len(X), len(self.classes_)))
        for cols, m in zip(self.subsets_, self.models_):
            pred = m.predict(X[:, cols])
            for i, c in enumerate(self.classes_):
                votes[:, i] += (pred == c)
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


__all__ = ["RandomSubspaceClassifier"]
