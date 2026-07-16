"""Missing-value imputation."""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, check_is_fitted, clone
from ..utils import check_random_state


def _to_float_with_nan(X, missing_values):
    X = np.array(X, dtype=np.float64, copy=True)
    if not (isinstance(missing_values, float) and np.isnan(missing_values)):
        X[X == missing_values] = np.nan
    return X


class SimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, missing_values=np.nan, strategy="mean", fill_value=None):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X, y=None):
        X = _to_float_with_nan(X, self.missing_values)
        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == "most_frequent":
            stats = np.empty(X.shape[1])
            for j in range(X.shape[1]):
                col = X[~np.isnan(X[:, j]), j]
                if len(col) == 0:
                    stats[j] = 0.0
                else:
                    vals, counts = np.unique(col, return_counts=True)
                    stats[j] = vals[np.argmax(counts)]
            self.statistics_ = stats
        elif self.strategy == "constant":
            self.statistics_ = np.full(X.shape[1],
                                       0.0 if self.fill_value is None
                                       else self.fill_value)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")
        return self

    def transform(self, X):
        check_is_fitted(self, "statistics_")
        X = _to_float_with_nan(X, self.missing_values)
        mask = np.isnan(X)
        X[mask] = np.take(self.statistics_, np.where(mask)[1])
        return X


class MissingIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, missing_values=np.nan, features="missing-only"):
        self.missing_values = missing_values
        self.features = features

    def fit(self, X, y=None):
        X = _to_float_with_nan(X, self.missing_values)
        mask = np.isnan(X)
        if self.features == "missing-only":
            self.features_ = np.where(mask.any(axis=0))[0]
        else:
            self.features_ = np.arange(X.shape[1])
        return self

    def transform(self, X):
        check_is_fitted(self, "features_")
        X = _to_float_with_nan(X, self.missing_values)
        return np.isnan(X)[:, self.features_]


class KNNImputer(BaseEstimator, TransformerMixin):
    """Impute from the k nearest neighbors under nan-euclidean distance."""

    def __init__(self, n_neighbors=5, weights="uniform", missing_values=np.nan):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.missing_values = missing_values

    def fit(self, X, y=None):
        self._fit_X = _to_float_with_nan(X, self.missing_values)
        self._col_means = np.nanmean(self._fit_X, axis=0)
        return self

    @staticmethod
    def _nan_euclidean(A, B):
        # scaled euclidean ignoring coordinates missing in either point
        d = np.empty((len(A), len(B)))
        for i, a in enumerate(A):
            valid = ~(np.isnan(a)[None, :] | np.isnan(B))
            diff = np.where(valid, a[None, :] - np.where(np.isnan(B), 0, B), 0.0)
            counts = valid.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                d[i] = np.sqrt(np.where(counts > 0,
                                        (diff ** 2).sum(axis=1)
                                        * (A.shape[1] / np.maximum(counts, 1)),
                                        np.inf))
        return d

    def transform(self, X):
        check_is_fitted(self, "_fit_X")
        X = _to_float_with_nan(X, self.missing_values)
        out = X.copy()
        rows_missing = np.where(np.isnan(X).any(axis=1))[0]
        if len(rows_missing) == 0:
            return out
        D = self._nan_euclidean(X[rows_missing], self._fit_X)
        for r, i in enumerate(rows_missing):
            for j in np.where(np.isnan(X[i]))[0]:
                donors_ok = ~np.isnan(self._fit_X[:, j])
                d = np.where(donors_ok, D[r], np.inf)
                order = np.argsort(d)[: self.n_neighbors]
                order = order[np.isfinite(d[order])]
                if len(order) == 0:
                    out[i, j] = self._col_means[j]
                    continue
                vals = self._fit_X[order, j]
                if self.weights == "distance":
                    w = 1.0 / np.maximum(d[order], 1e-12)
                    out[i, j] = (vals * w).sum() / w.sum()
                else:
                    out[i, j] = vals.mean()
        return out


class IterativeImputer(BaseEstimator, TransformerMixin):
    """Round-robin regression imputation (MICE-style, ridge by default)."""

    def __init__(self, estimator=None, max_iter=10, tol=1e-3,
                 missing_values=np.nan, random_state=None):
        self.estimator = estimator
        self.max_iter = max_iter
        self.tol = tol
        self.missing_values = missing_values
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        from ..linear_model import Ridge
        X = _to_float_with_nan(X, self.missing_values)
        mask = np.isnan(X)
        self._col_means = np.nanmean(X, axis=0)
        Xf = X.copy()
        idx = np.where(mask)
        Xf[idx] = np.take(self._col_means, idx[1])
        base = self.estimator if self.estimator is not None else Ridge(alpha=1e-3)
        cols = np.where(mask.any(axis=0))[0]
        self.estimators_ = {}
        prev = Xf[mask].copy() if mask.any() else np.array([])
        for it in range(self.max_iter):
            for j in cols:
                obs = ~mask[:, j]
                other = np.delete(np.arange(X.shape[1]), j)
                est = clone(base).fit(Xf[obs][:, other], Xf[obs, j])
                self.estimators_[j] = (est, other)
                if (~obs).any():
                    Xf[~obs, j] = est.predict(Xf[~obs][:, other])
            if mask.any():
                cur = Xf[mask]
                change = np.max(np.abs(cur - prev)) / max(np.nanstd(X), 1e-12)
                prev = cur.copy()
                if change < self.tol:
                    break
        self.n_iter_ = it + 1
        return Xf

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "estimators_")
        X = _to_float_with_nan(X, self.missing_values)
        mask = np.isnan(X)
        Xf = X.copy()
        idx = np.where(mask)
        Xf[idx] = np.take(self._col_means, idx[1])
        for j, (est, other) in self.estimators_.items():
            miss = mask[:, j]
            if miss.any():
                Xf[miss, j] = est.predict(Xf[miss][:, other])
        return Xf


__all__ = ["SimpleImputer", "MissingIndicator", "KNNImputer", "IterativeImputer"]
