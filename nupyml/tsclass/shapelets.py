"""Shapelets: classify by distance to discriminative SUBSEQUENCES.

THE IDEA
--------
A shapelet is a short subsequence -- a shape -- that is characteristic of a
class. The shapelet feature of a series is the distance from that series to its
CLOSEST-matching window: "how nearly does this series contain shape X, anywhere
in it?". Because the match slides over every position, the feature is invariant
to WHERE the shape occurs, and because a shapelet is an actual piece of a real
series, the resulting model is interpretable -- you can plot the shape that made
the decision.

THIS IMPLEMENTATION
-------------------
The random shapelet transform: sample many candidate subsequences from the
training set, use each as a feature (z-normalised sliding-window minimum
distance), and hand the resulting tabular matrix to any classifier. Sampling
random shapelets, rather than searching for the single most discriminative one,
is what made the shapelet transform fast enough to be practical.

Distances are computed on Z-NORMALISED windows so that a shape is matched by its
form, not its offset or scale -- the standard choice for shapelets.

Lines et al. (2012); Hills et al. (2014).
"""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, ClassifierMixin, check_is_fitted
from ..utils import check_array, check_X_y, check_random_state


def _znorm(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else x - x.mean()


def _min_dist(series, shapelet):
    """Minimum z-normalised Euclidean distance from a shapelet to any window."""
    m = len(shapelet)
    sz = _znorm(shapelet)
    best = np.inf
    for start in range(len(series) - m + 1):
        w = _znorm(series[start:start + m])
        d = np.sqrt(np.sum((w - sz) ** 2))
        if d < best:
            best = d
    return best


class ShapeletTransform(BaseEstimator, TransformerMixin):
    """Turn each series into a vector of distances to sampled shapelets.

    ``fit`` draws ``n_shapelets`` random subsequences (each of a random length in
    ``[min_length, max_length]``) from random training series and keeps them;
    ``transform`` computes the sliding-window minimum distance from every series
    to every shapelet.
    """

    def __init__(self, n_shapelets=100, min_length=5, max_length=None,
                 random_state=None):
        self.n_shapelets = n_shapelets
        self.min_length = min_length
        self.max_length = max_length
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        L = X.shape[1]
        max_len = self.max_length or max(self.min_length + 1, L // 2)
        self.shapelets_ = []
        for _ in range(self.n_shapelets):
            m = int(rng.randint(self.min_length, max_len + 1))
            i = rng.randint(len(X))
            start = rng.randint(0, L - m + 1)
            self.shapelets_.append(X[i, start:start + m].copy())
        return self

    def transform(self, X):
        check_is_fitted(self, "shapelets_")
        X = check_array(X)
        out = np.empty((len(X), len(self.shapelets_)))
        for i, series in enumerate(X):
            for j, sh in enumerate(self.shapelets_):
                out[i, j] = _min_dist(series, sh)
        return out


class ShapeletTransformClassifier(BaseEstimator, ClassifierMixin):
    """Shapelet-distance features + a random forest."""

    def __init__(self, n_shapelets=100, min_length=5, max_length=None,
                 n_estimators=100, random_state=None):
        self.n_shapelets = n_shapelets
        self.min_length = min_length
        self.max_length = max_length
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        from ..ensemble import RandomForestClassifier
        X, y = check_X_y(X, y)
        self.transform_ = ShapeletTransform(
            self.n_shapelets, self.min_length, self.max_length,
            self.random_state).fit(X)
        F = self.transform_.transform(X)
        self.clf_ = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state).fit(F, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self, "clf_")
        return self.clf_.predict(self.transform_.transform(check_array(X)))


__all__ = ["ShapeletTransform", "ShapeletTransformClassifier"]
