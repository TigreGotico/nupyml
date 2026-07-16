"""Isotonic (monotone) regression via pool-adjacent-violators."""
import numpy as np

from ..base import BaseEstimator, RegressorMixin, TransformerMixin, check_is_fitted
from ..utils import column_or_1d


def isotonic_regression(y, sample_weight=None, increasing=True):
    """Pool-adjacent-violators: the L2-closest monotone fit to ``y``."""
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(y)) if sample_weight is None \
        else np.asarray(sample_weight, dtype=np.float64).copy()
    if not increasing:
        y = -y
    # active-set PAVA: blocks of (weighted mean, weight, size)
    values, weights, sizes = [], [], []
    for yi, wi in zip(y, w):
        values.append(yi)
        weights.append(wi)
        sizes.append(1)
        # merge while the last block violates monotonicity
        while len(values) > 1 and values[-2] > values[-1]:
            v2, w2, s2 = values.pop(), weights.pop(), sizes.pop()
            v1, w1, s1 = values.pop(), weights.pop(), sizes.pop()
            total_w = w1 + w2
            values.append((v1 * w1 + v2 * w2) / total_w)
            weights.append(total_w)
            sizes.append(s1 + s2)
    out = np.repeat(values, sizes)
    return -out if not increasing else out


class IsotonicRegression(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, y_min=None, y_max=None, increasing=True,
                 out_of_bounds="clip"):
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

    def fit(self, X, y, sample_weight=None):
        X = column_or_1d(X).astype(np.float64)
        y = column_or_1d(y).astype(np.float64)
        w = np.ones(len(y)) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        increasing = self.increasing
        if increasing == "auto":
            increasing = np.corrcoef(X, y)[0, 1] >= 0
        order = np.argsort(X, kind="stable")
        Xs, ys, ws = X[order], y[order], w[order]
        # average ties so the fit is a function of x
        uniq, inv = np.unique(Xs, return_inverse=True)
        if len(uniq) < len(Xs):
            wsum = np.bincount(inv, weights=ws)
            ysum = np.bincount(inv, weights=ws * ys)
            Xs, ys, ws = uniq, ysum / wsum, wsum
        fitted = isotonic_regression(ys, ws, increasing=increasing)
        if self.y_min is not None:
            fitted = np.maximum(fitted, self.y_min)
        if self.y_max is not None:
            fitted = np.minimum(fitted, self.y_max)
        self.X_thresholds_ = Xs
        self.y_thresholds_ = fitted
        self.increasing_ = increasing
        self.X_min_, self.X_max_ = float(Xs[0]), float(Xs[-1])
        return self

    def predict(self, X):
        check_is_fitted(self, "X_thresholds_")
        X = column_or_1d(X).astype(np.float64)
        if self.out_of_bounds == "raise" and (
                (X < self.X_min_).any() or (X > self.X_max_).any()):
            raise ValueError("X contains values outside the training range")
        return np.interp(np.clip(X, self.X_min_, self.X_max_),
                         self.X_thresholds_, self.y_thresholds_)

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y=None, **kw):
        return self.fit(X, y, **kw).transform(X)


__all__ = ["IsotonicRegression", "isotonic_regression"]
