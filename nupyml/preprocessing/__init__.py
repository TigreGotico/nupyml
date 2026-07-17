"""Preprocessing: putting data into a form a model can use.

WHY SCALING MATTERS (AND WHEN IT DOES NOT)
------------------------------------------
Any model that measures DISTANCE or penalises COEFFICIENTS is at the mercy of
units. If one feature is in metres and another in kilometres, the first dominates
every distance and the ridge penalty punishes the second's larger coefficients
for no reason but its units. So kNN, SVM, KMeans, PCA and every penalised linear
model need scaled input.

Trees do not. They only compare a feature to a threshold, so any monotone
rescaling leaves the tree identical. Scaling before a random forest is harmless
and pointless.

WHICH SCALER
------------
* ``StandardScaler`` -- subtract mean, divide by std. The default. Assumes
  roughly symmetric data; a single wild outlier distorts both statistics.
* ``RobustScaler`` -- uses median and IQR, which outliers cannot move. Use it
  when they exist and are real.
* ``MinMaxScaler`` -- squashes into [0, 1]. Preserves the shape exactly, but a
  single extreme value compresses everything else into a sliver.
* ``Normalizer`` -- scales each ROW to unit norm, not each column. A different
  operation for a different purpose: it makes direction matter and magnitude
  not, which is what text similarity usually wants.

ENCODING CATEGORIES
-------------------
``OrdinalEncoder`` maps categories to 0, 1, 2... which INVENTS an order: a model
will conclude that "blue" (2) is greater than "red" (1), and halfway between it
and "green" (3). Fine for a tree, actively wrong for anything linear or
distance-based.

``OneHotEncoder`` gives each category its own column, asserting no order. The
cost is width, and correlated columns (they sum to 1).

FIT ON TRAIN, TRANSFORM ON BOTH
-------------------------------
Every transformer here learns from data: a mean, a range, a category list. Those
must be learned from the TRAINING data only and applied unchanged to test data.
Fitting a scaler on everything leaks test statistics into training. This is why
``fit`` and ``transform`` are separate, and why ``Pipeline`` exists.
"""
import itertools

import numpy as np

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_array, column_or_1d


class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1])
        if self.with_std:
            scale = X.std(axis=0)
            scale[scale == 0.0] = 1.0
            self.scale_ = scale
        else:
            self.scale_ = np.ones(X.shape[1])
        return self

    def partial_fit(self, X, y=None):
        """Update mean/variance from a new chunk (Welford-style)."""
        X = check_array(X)
        if not hasattr(self, "n_samples_seen_"):
            self.n_samples_seen_ = 0
            self._sum = np.zeros(X.shape[1])
            self._sq_sum = np.zeros(X.shape[1])
        self.n_samples_seen_ += len(X)
        self._sum += X.sum(axis=0)
        self._sq_sum += (X ** 2).sum(axis=0)
        mean = self._sum / self.n_samples_seen_
        var = np.maximum(self._sq_sum / self.n_samples_seen_ - mean ** 2, 0.0)
        self.mean_ = mean if self.with_mean else np.zeros(X.shape[1])
        if self.with_std:
            scale = np.sqrt(var)
            scale[scale == 0.0] = 1.0
            self.scale_ = scale
        else:
            self.scale_ = np.ones(X.shape[1])
        self.var_ = var
        return self

    def transform(self, X):
        check_is_fitted(self, "mean_")
        X = check_array(X)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        check_is_fitted(self, "mean_")
        return np.asarray(X) * self.scale_ + self.mean_


class MinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        X = check_array(X)
        lo, hi = self.feature_range
        data_min = X.min(axis=0)
        data_max = X.max(axis=0)
        rng = data_max - data_min
        rng[rng == 0.0] = 1.0
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.scale_ = (hi - lo) / rng
        self.min_ = lo - data_min * self.scale_
        return self

    def partial_fit(self, X, y=None):
        X = check_array(X)
        lo, hi = self.feature_range
        data_min = X.min(axis=0)
        data_max = X.max(axis=0)
        if hasattr(self, "data_min_"):
            data_min = np.minimum(data_min, self.data_min_)
            data_max = np.maximum(data_max, self.data_max_)
        rng = data_max - data_min
        rng[rng == 0.0] = 1.0
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.scale_ = (hi - lo) / rng
        self.min_ = lo - data_min * self.scale_
        return self

    def transform(self, X):
        check_is_fitted(self, "scale_")
        return check_array(X) * self.scale_ + self.min_

    def inverse_transform(self, X):
        check_is_fitted(self, "scale_")
        return (np.asarray(X) - self.min_) / self.scale_


class MaxAbsScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = check_array(X)
        scale = np.abs(X).max(axis=0)
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X):
        check_is_fitted(self, "scale_")
        return check_array(X) / self.scale_


class RobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self, quantile_range=(25.0, 75.0)):
        self.quantile_range = quantile_range

    def fit(self, X, y=None):
        X = check_array(X)
        q_lo, q_hi = self.quantile_range
        self.center_ = np.median(X, axis=0)
        scale = np.percentile(X, q_hi, axis=0) - np.percentile(X, q_lo, axis=0)
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X):
        check_is_fitted(self, "center_")
        return (check_array(X) - self.center_) / self.scale_


class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self, norm="l2"):
        self.norm = norm

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = check_array(X)
        if self.norm == "l2":
            norms = np.linalg.norm(X, axis=1)
        elif self.norm == "l1":
            norms = np.abs(X).sum(axis=1)
        elif self.norm == "max":
            norms = np.abs(X).max(axis=1)
        else:
            raise ValueError(f"Unknown norm: {self.norm!r}")
        norms[norms == 0.0] = 1.0
        return X / norms[:, None]


class LabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, y):
        self.classes_ = np.unique(column_or_1d(y))
        return self

    def transform(self, y):
        check_is_fitted(self, "classes_")
        y = column_or_1d(y)
        idx = np.searchsorted(self.classes_, y)
        bad = (idx >= len(self.classes_)) | (self.classes_[np.minimum(idx, len(self.classes_) - 1)] != y)
        if np.any(bad):
            raise ValueError(f"y contains previously unseen labels: {np.unique(y[bad])}")
        return idx

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        check_is_fitted(self, "classes_")
        return self.classes_[np.asarray(y, dtype=int)]


class LabelBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, y):
        self.classes_ = np.unique(column_or_1d(y))
        return self

    def transform(self, y):
        check_is_fitted(self, "classes_")
        y = column_or_1d(y)
        if len(self.classes_) == 2:
            return (y == self.classes_[1]).astype(np.float64)[:, None]
        return (y[:, None] == self.classes_[None, :]).astype(np.float64)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, Y):
        check_is_fitted(self, "classes_")
        Y = np.asarray(Y)
        if Y.ndim == 2 and Y.shape[1] == 1:
            return self.classes_[(Y.ravel() > 0.5).astype(int)]
        return self.classes_[np.argmax(Y, axis=1)]


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, handle_unknown="error"):
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        check_is_fitted(self, "categories_")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        cols = []
        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            onehot = (col[:, None] == cats[None, :]).astype(np.float64)
            if self.handle_unknown == "error" and not np.all(onehot.sum(axis=1) == 1):
                raise ValueError(f"Found unknown categories in column {j}")
            cols.append(onehot)
        return np.hstack(cols)


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        check_is_fitted(self, "categories_")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        out = np.empty(X.shape, dtype=np.float64)
        for j, cats in enumerate(self.categories_):
            out[:, j] = np.searchsorted(cats, X[:, j])
        return out


class PolynomialFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, interaction_only=False, include_bias=True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def _combinations(self, n_features):
        comb = itertools.combinations if self.interaction_only else \
            itertools.combinations_with_replacement
        start = 0 if self.include_bias else 1
        for d in range(start, self.degree + 1):
            yield from comb(range(n_features), d)

    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        self.n_output_features_ = sum(1 for _ in self._combinations(X.shape[1]))
        self.n_features_out_ = self.n_output_features_
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        X = check_array(X)
        cols = [np.prod(X[:, c], axis=1) if c else np.ones(len(X))
                for c in self._combinations(self.n_features_in_)]
        return np.column_stack(cols)


class Binarizer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (check_array(X) > self.threshold).astype(np.float64)


class KBinsDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=5, strategy="quantile"):
        self.n_bins = n_bins
        self.strategy = strategy

    def fit(self, X, y=None):
        X = check_array(X)
        edges = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if self.strategy == "quantile":
                q = np.linspace(0, 100, self.n_bins + 1)
                e = np.percentile(col, q)
            elif self.strategy == "uniform":
                e = np.linspace(col.min(), col.max(), self.n_bins + 1)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy!r}")
            edges.append(np.unique(e))
        self.bin_edges_ = edges
        return self

    def transform(self, X):
        check_is_fitted(self, "bin_edges_")
        X = check_array(X)
        out = np.empty(X.shape, dtype=np.float64)
        for j, e in enumerate(self.bin_edges_):
            out[:, j] = np.clip(np.searchsorted(e[1:-1], X[:, j], side="right"),
                                0, len(e) - 2)
        return out


__all__ = [
    "StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler",
    "Normalizer", "LabelEncoder", "LabelBinarizer", "OneHotEncoder",
    "OrdinalEncoder", "PolynomialFeatures", "Binarizer", "KBinsDiscretizer",
]
