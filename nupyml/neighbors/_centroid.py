"""Nearest-centroid classification and radius-based neighbours."""
import numpy as np
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, ClassifierMixin, check_is_fitted
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array


class NearestCentroid(BaseEstimator, ClassifierMixin):
    """Represent each class by ONE point -- its centroid -- and classify by nearest.

    THE SIMPLEST DISTANCE CLASSIFIER
    --------------------------------
    Reduce each class to the mean of its members, then label a new point by which
    centroid is closest. No neighbours to store, no k to tune, one vector per
    class -- it is kNN's minimalist cousin, and it is exactly the decision rule of
    Gaussian LDA when the classes share a spherical covariance.

    WHERE IT WORKS AND WHERE IT FAILS
    ---------------------------------
    Because a class is a single point, it assumes each class is a roughly convex,
    unimodal blob. That holds surprisingly often for high-dimensional data (it is
    the "Rocchio" text classifier), and it is fast and robust when it does. It
    fails hard when a class is multi-modal or wraps around another -- two
    well-separated clusters of the same class average to a centroid sitting between
    them, in the wrong region entirely. Knowing that failure mode is the point of
    seeing it next to kNN.

    THE SHRINKAGE OPTION
    --------------------
    ``shrink_threshold`` shrinks each class centroid toward the overall centroid,
    zeroing the features where the class barely differs from the mean. This does
    feature selection FOR FREE -- noisy features that do not separate the classes
    are removed -- and is the "nearest shrunken centroid" method used for gene
    expression, where most genes are irrelevant.

    Tibshirani et al. (2002).
    """

    def __init__(self, metric="euclidean", shrink_threshold=None):
        self.metric = metric
        self.shrink_threshold = shrink_threshold

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        y_idx = self._le.transform(y)

        self.centroids_ = np.array([X[y_idx == c].mean(axis=0)
                                    for c in range(len(self.classes_))])
        if self.shrink_threshold is not None:
            self._shrink(X, y_idx)
        return self

    def _shrink(self, X, y_idx):
        """Shrink each centroid toward the global one, thresholding small
        deviations to zero -- nearest shrunken centroids."""
        overall = X.mean(axis=0)
        within_std = X.std(axis=0) + 1e-8
        for c in range(len(self.classes_)):
            nc = np.sum(y_idx == c)
            # standardised deviation of the class centroid from the global one
            dev = (self.centroids_[c] - overall) / (within_std / np.sqrt(nc))
            # soft-threshold: features where the class barely differs are zeroed
            dev = np.sign(dev) * np.maximum(np.abs(dev) - self.shrink_threshold, 0)
            self.centroids_[c] = overall + dev * within_std / np.sqrt(nc)

    def predict(self, X):
        check_is_fitted(self, "centroids_")
        d = cdist(check_array(X), self.centroids_, metric=self.metric)
        return self.classes_[np.argmin(d, axis=1)]


class RadiusNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """Classify by ALL neighbours within a fixed RADIUS, not a fixed count.

    THE DIFFERENCE FROM kNN
    -----------------------
    kNN always uses exactly ``k`` neighbours, however near or far. In a dense
    region those k are all close and trustworthy; in a sparse region they may be
    far away and irrelevant, yet kNN uses them anyway. RadiusNeighbors instead
    uses EVERY point within ``radius`` -- many votes in dense regions, few (or
    none) in sparse ones.

    That adapts the neighbourhood to the local density, which is more honest where
    density varies wildly. The price is the failure mode kNN cannot have: a query
    in an empty region may have NO neighbours within the radius, and then there is
    nothing to vote -- ``outlier_label`` decides what to do, and choosing the
    radius is genuinely harder than choosing k because it lives in the data's
    units, not its counts.
    """

    def __init__(self, radius=1.0, outlier_label=None, metric="euclidean"):
        self.radius = radius
        self.outlier_label = outlier_label
        self.metric = metric

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self._X = X
        self._y = self._le.transform(y)
        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X)
        d = cdist(X, self._X, metric=self.metric)
        out = np.empty(len(X), dtype=int)
        for i in range(len(X)):
            within = np.where(d[i] <= self.radius)[0]
            if len(within) == 0:
                # the empty-neighbourhood case radius search must handle
                if self.outlier_label is None:
                    out[i] = np.bincount(self._y).argmax()   # fall back to prior
                else:
                    out[i] = self._le.transform([self.outlier_label])[0]
            else:
                out[i] = np.bincount(self._y[within]).argmax()
        return self.classes_[out]


__all__ = ["NearestCentroid", "RadiusNeighborsClassifier"]
