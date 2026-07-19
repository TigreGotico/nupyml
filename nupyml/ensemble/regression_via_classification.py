"""Regression via classification: discretise the target, classify, decode."""
import numpy as np

from ..base import BaseEstimator, RegressorMixin, clone
from ..tree import DecisionTreeClassifier
from ..utils import check_X_y, check_array


class RegressionViaClassification(BaseEstimator, RegressorMixin):
    """Turn regression into classification over BINS (Torgo & Gama, 1997).

    Some targets are easier to CLASSIFY than to regress -- a classifier can carve a
    non-monotone, multi-modal response that a single regressor smooths over. This
    discretises the target into bins, trains a classifier to predict the bin, and
    decodes back to a number as the class-probability-weighted average of the bin
    centres. The soft decode keeps the estimate continuous while borrowing the
    classifier's flexible decision boundaries. ``n_bins`` sets the granularity;
    ``strategy`` is 'quantile' (equal counts) or 'uniform' (equal width).
    """

    def __init__(self, classifier=None, n_bins=10, strategy="quantile"):
        self.classifier = classifier
        self.n_bins = n_bins
        self.strategy = strategy

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.strategy == "quantile":
            self.edges_ = np.quantile(y, np.linspace(0, 1, self.n_bins + 1))
            self.edges_ = np.unique(self.edges_)
        else:
            self.edges_ = np.linspace(y.min(), y.max(), self.n_bins + 1)
        bins = np.clip(np.digitize(y, self.edges_[1:-1]), 0, len(self.edges_) - 2)
        self.centres_ = np.array([y[bins == b].mean() if np.any(bins == b)
                                  else 0.5 * (self.edges_[b] + self.edges_[b + 1])
                                  for b in range(len(self.edges_) - 1)])
        clf = self.classifier
        if clf is None:
            clf = DecisionTreeClassifier(max_depth=6)
        self.clf_ = clone(clf).fit(X, bins)
        self.bin_labels_ = self.clf_.classes_
        return self

    def predict(self, X):
        X = check_array(X)
        proba = self.clf_.predict_proba(X)
        centres = self.centres_[self.bin_labels_]
        return proba @ centres                             # soft-decoded expectation


__all__ = ["RegressionViaClassification"]
