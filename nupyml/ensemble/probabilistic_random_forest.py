"""Probabilistic random forest: a forest that propagates feature uncertainty."""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin
from ..tree import DecisionTreeClassifier
from ..utils import check_X_y, check_array, check_random_state


class ProbabilisticRandomForest(BaseEstimator, ClassifierMixin):
    """A forest that PROPAGATES feature uncertainty (Reis et al., 2019).

    A normal forest treats every measurement as exact, but real features come with
    error bars (a noisy sensor, a blurred photometry). The probabilistic random forest
    treats each feature value as a GAUSSIAN and, at prediction time, sends a sample
    down each tree drawn from that per-feature distribution -- repeated over many draws
    and many trees, so the vote reflects both the forest's disagreement AND the input's
    own uncertainty. Fitting is ordinary bagging; the probabilistic part is the
    sampled, uncertainty-aware inference. ``feature_sigma`` is the per-feature noise.
    """

    def __init__(self, n_estimators=25, max_depth=None, n_samples=10,
                 feature_sigma=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_samples = n_samples
        self.feature_sigma = feature_sigma
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        self.classes_ = np.unique(y)
        n = len(X)
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, n, n)                     # bootstrap sample
            t = DecisionTreeClassifier(max_depth=self.max_depth,
                                       random_state=rng.randint(1 << 30))
            t.fit(X[idx], y[idx])
            self.trees_.append(t)
        self._rng_seed = rng.randint(1 << 30)
        return self

    def predict_proba(self, X):
        X = check_array(X)
        rng = check_random_state(self._rng_seed)
        sigma = np.atleast_1d(self.feature_sigma)
        votes = np.zeros((len(X), len(self.classes_)))
        for t in self.trees_:
            for _ in range(self.n_samples):
                # perturb inputs by their measurement noise, then vote
                Xs = X + rng.randn(*X.shape) * sigma
                for i, c in enumerate(t.predict(Xs)):
                    votes[i, np.searchsorted(self.classes_, c)] += 1
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


__all__ = ["ProbabilisticRandomForest"]
