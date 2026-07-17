"""Explainable Boosting Machine (EBM / GA2M): a glass box that is also accurate.

THE TENSION IT RESOLVES
-----------------------
Linear models are interpretable but too rigid; gradient boosting is accurate but
opaque. A Generalised Additive Model splits the difference::

    g(E[y]) = intercept + f_1(x_1) + f_2(x_2) + ... + f_d(x_d)

-- each feature gets its OWN arbitrary shape function ``f_j``, and the prediction
is just their sum. You can PLOT every ``f_j`` and read exactly how the model uses
each feature; there is no interaction to hide a surprise. EBM (GA2M) adds a few
pairwise terms ``f_{jk}(x_j, x_k)`` for the interactions that matter, recovering
most of a full GBM's accuracy while staying additive and readable.

HOW IT IS FIT
-------------
Round-robin gradient boosting, one feature at a time: bin each feature, and
repeatedly fit a tiny tree to the residual using ONLY that feature, cycling
through the features with a low learning rate. Cycling (rather than letting boost
pick the best feature greedily) is what keeps the effects DISENTANGLED -- each
feature's shape absorbs only its own marginal effect, so the plots are honest.
The learned shape of feature ``j`` is the summed contribution per bin, exposed in
``shape_functions_``.

Lou, Caruana & Gehrke (2012, 2013); Nori et al. (2019).
"""
import numpy as np

from ..base import BaseEstimator, RegressorMixin, ClassifierMixin, check_is_fitted
from ..utils import check_X_y, check_array


class _BaseEBM(BaseEstimator):
    def __init__(self, n_bins=32, n_rounds=200, learning_rate=0.1, random_state=None):
        self.n_bins = n_bins
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _bin(self, X):
        # per-feature quantile bin edges, learned at fit time
        idx = np.empty(X.shape, dtype=int)
        for j in range(X.shape[1]):
            e = self.bin_edges_[j]
            idx[:, j] = np.clip(np.searchsorted(e, X[:, j]), 0, len(e))
        return idx

    def _fit_shapes(self, X, residual_target):
        """Cyclic boosting of per-bin additive contributions on a target."""
        n, d = X.shape
        self.bin_edges_ = []
        for j in range(d):
            qs = np.linspace(0, 100, self.n_bins + 1)[1:-1]
            self.bin_edges_.append(np.percentile(X[:, j], qs))
        binned = self._bin(X)
        n_bins_actual = [len(e) + 1 for e in self.bin_edges_]
        shapes = [np.zeros(nb) for nb in n_bins_actual]
        lr = self.learning_rate

        pred = np.full(n, self.intercept_, dtype=float)
        for _ in range(self.n_rounds):
            grad = self._negative_gradient(residual_target, pred)
            for j in range(d):
                # best per-bin constant step on this feature's gradient
                b = binned[:, j]
                update = np.zeros(n_bins_actual[j])
                counts = np.bincount(b, minlength=n_bins_actual[j])
                sums = np.bincount(b, weights=grad, minlength=n_bins_actual[j])
                nz = counts > 0
                update[nz] = lr * sums[nz] / counts[nz]
                shapes[j] += update
                pred += update[b]
        self.shape_functions_ = shapes
        return binned

    def _contributions(self, X):
        binned = self._bin(check_array(X))
        total = np.full(len(binned), self.intercept_, dtype=float)
        for j in range(binned.shape[1]):
            total += self.shape_functions_[j][binned[:, j]]
        return total

    def explain_global(self, feature):
        """Return (bin_edges, per-bin contribution) -- the plottable shape."""
        check_is_fitted(self, "shape_functions_")
        return self.bin_edges_[feature], self.shape_functions_[feature]


class ExplainableBoostingRegressor(_BaseEBM, RegressorMixin):
    """EBM for regression (squared-error, so the gradient is the residual)."""

    def _negative_gradient(self, y, pred):
        return y - pred

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.intercept_ = float(np.mean(y))
        self._fit_shapes(X, y)
        return self

    def predict(self, X):
        return self._contributions(X)


class ExplainableBoostingClassifier(_BaseEBM, ClassifierMixin):
    """EBM for binary classification (logistic loss; shapes are in log-odds)."""

    def _negative_gradient(self, y, logit):
        p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        return y - p                                # logistic gradient

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        yb = (y == self.classes_[-1]).astype(float)
        base = yb.mean()
        self.intercept_ = float(np.log((base + 1e-6) / (1 - base + 1e-6)))
        self._fit_shapes(X, yb)
        return self

    def predict_proba(self, X):
        logit = self._contributions(X)
        p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


__all__ = ["ExplainableBoostingRegressor", "ExplainableBoostingClassifier"]
