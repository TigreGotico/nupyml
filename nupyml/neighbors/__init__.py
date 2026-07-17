"""Nearest neighbours: predict by looking up similar examples.

THE LAZIEST POSSIBLE MODEL
--------------------------
There is no training. ``fit`` stores the data; all the work happens at predict
time, when the k closest training points vote (classification) or average
(regression). The "model" IS the dataset.

That makes it non-parametric in the strict sense: the decision boundary can be
arbitrarily complicated, because it is never summarised into parameters. With
k=1 the training error is exactly zero -- every point is its own neighbour --
which tells you nothing about generalisation. k controls the smoothing: small k
is a jagged, low-bias, high-variance boundary; large k averages over a wide
neighbourhood and eventually predicts the global majority.

WHAT IT COSTS
-------------
* Prediction is O(n) per query in the worst case; a KD-tree cuts that to
  O(log n) in low dimensions, which is why ``cKDTree`` is used here.
* The whole training set must be kept.
* Distance is meaningless unless features are scaled. A feature measured in
  metres will dominate one measured in kilometres, purely through units.

THE CURSE OF DIMENSIONALITY
---------------------------
This is the real limit, and it is worse than it sounds. As dimensions grow,
volume grows exponentially, so any fixed number of samples becomes hopelessly
sparse -- and the distances between points CONCENTRATE: the nearest and
farthest neighbours end up almost equidistant. Once that happens "nearest"
carries no information, and kNN degrades to guessing while looking like it is
working. KD-trees degrade to brute force at the same time, for the same reason.

Reducing the dimension first (``nupyml.decomposition``) is often what makes kNN
viable at all.
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.special import logsumexp

from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                    DensityMixin, check_is_fitted)
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array


def _densify(X):
    """cKDTree needs dense coordinates, so sparse input is expanded once."""
    X = check_array(X, accept_sparse=True)
    return np.asarray(X.todense()) if sp.issparse(X) else X


class _KNeighborsBase(BaseEstimator):
    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def _kneighbors(self, X):
        dist, idx = self._tree.query(X, k=self.n_neighbors)
        if self.n_neighbors == 1:
            dist, idx = dist[:, None], idx[:, None]
        return dist, idx

    def _weights_for(self, dist):
        if self.weights == "uniform":
            return np.ones_like(dist)
        if self.weights == "distance":
            with np.errstate(divide="ignore"):
                w = 1.0 / dist
            # exact matches get all the weight
            inf = np.isinf(w)
            rows = inf.any(axis=1)
            w[rows] = inf[rows].astype(float)
            return w
        raise ValueError(f"Unknown weights: {self.weights!r}")

    def kneighbors(self, X, n_neighbors=None):
        check_is_fitted(self, "_tree")
        X = check_array(X)
        k = n_neighbors or self.n_neighbors
        dist, idx = self._tree.query(X, k=k)
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]
        return dist, idx


class KNeighborsClassifier(_KNeighborsBase, ClassifierMixin):
    """Classify by majority vote among the k nearest training points.

    ``weights="uniform"`` gives every neighbour an equal vote.
    ``weights="distance"`` weights by ``1/d``, so closer neighbours count more
    -- which also means an exact match dominates completely (its weight is
    infinite, handled explicitly), and is why distance weighting reproduces the
    training labels perfectly.
    """

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        X = _densify(X)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self._y = self._le.transform(y)
        self._tree = cKDTree(X)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_tree")
        X = _densify(X)
        dist, idx = self._kneighbors(X)
        w = self._weights_for(dist)
        k = len(self.classes_)
        proba = np.zeros((len(X), k))
        labels = self._y[idx]
        for c in range(k):
            proba[:, c] = np.where(labels == c, w, 0.0).sum(axis=1)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class KNeighborsRegressor(_KNeighborsBase, RegressorMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True, accept_sparse=True)
        self._y = y
        self._tree = cKDTree(_densify(X))
        return self

    def predict(self, X):
        check_is_fitted(self, "_tree")
        X = _densify(X)
        dist, idx = self._kneighbors(X)
        w = self._weights_for(dist)
        return (self._y[idx] * w).sum(axis=1) / w.sum(axis=1)


class NearestNeighbors(BaseEstimator):
    def __init__(self, n_neighbors=5, radius=1.0):
        self.n_neighbors = n_neighbors
        self.radius = radius

    def fit(self, X, y=None):
        self._X = _densify(X)
        self._tree = cKDTree(self._X)
        return self

    def kneighbors(self, X=None, n_neighbors=None):
        check_is_fitted(self, "_tree")
        k = n_neighbors or self.n_neighbors
        if X is None:
            dist, idx = self._tree.query(self._X, k=k + 1)
            return dist[:, 1:], idx[:, 1:]
        dist, idx = self._tree.query(check_array(X), k=k)
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]
        return dist, idx

    def radius_neighbors(self, X=None, radius=None):
        check_is_fitted(self, "_tree")
        r = radius or self.radius
        Xq = self._X if X is None else check_array(X)
        idx = self._tree.query_ball_point(Xq, r)
        return [np.asarray(i, dtype=int) for i in idx]


class KernelDensity(BaseEstimator, DensityMixin):
    """Estimate a probability density without assuming its shape.

    A histogram estimates a density but is blocky and depends on where the bin
    edges happen to fall. Kernel density estimation fixes both by putting a
    small smooth bump (the kernel) on top of EVERY data point and adding them
    up. The result is smooth and shift-invariant.

    ``bandwidth`` is the whole game -- far more important than which kernel is
    used. Too small and the estimate is a spike per sample (fitting noise); too
    large and every feature of the distribution is smoothed into one blob. It is
    the same bias-variance dial as k in kNN, and the same as ``bandwidth`` in
    MeanShift, which is really this density's modes.

    Being a density estimate rather than a classifier, it supports ``sample``:
    pick a training point at random, then jitter it by the kernel.
    """

    def __init__(self, bandwidth=1.0, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y=None):
        self._X = check_array(X)
        self._tree = cKDTree(self._X)
        return self

    def score_samples(self, X):
        check_is_fitted(self, "_tree")
        X = check_array(X)
        n, d = self._X.shape
        h = self.bandwidth
        if self.kernel == "gaussian":
            # exact: log mean of gaussian kernels
            diff = X[:, None, :] - self._X[None, :, :]
            sq = (diff ** 2).sum(axis=-1) / (2 * h * h)
            log_norm = -0.5 * d * np.log(2 * np.pi * h * h)
            return logsumexp(-sq, axis=1) + log_norm - np.log(n)
        if self.kernel == "tophat":
            import math
            counts = np.array([len(i) for i in self._tree.query_ball_point(X, h)])
            log_vol = ((d / 2) * np.log(np.pi) + d * np.log(h)
                       - math.lgamma(d / 2 + 1))
            with np.errstate(divide="ignore"):
                return np.log(counts / n) - log_vol
        raise ValueError(f"Unknown kernel: {self.kernel!r}")

    def score(self, X, y=None):
        return float(self.score_samples(X).sum())

    def sample(self, n_samples=1, random_state=None):
        from ..utils import check_random_state
        check_is_fitted(self, "_tree")
        rng = check_random_state(random_state)
        idx = rng.randint(0, len(self._X), size=n_samples)
        if self.kernel != "gaussian":
            raise NotImplementedError("sampling implemented for gaussian kernel")
        return self._X[idx] + rng.normal(scale=self.bandwidth,
                                         size=(n_samples, self._X.shape[1]))


__all__ = ["KNeighborsClassifier", "KNeighborsRegressor", "NearestNeighbors",
           "KernelDensity"]
