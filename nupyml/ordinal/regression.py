"""Ordinal regression: targets with an ORDER but no meaningful spacing.

THE IN-BETWEEN PROBLEM
----------------------
"poor < fair < good < excellent" is neither a nominal class (a classifier throws
away the order -- it treats "poor vs excellent" as no worse an error than "good vs
excellent") nor a real number (regression assumes the gaps are equal and
meaningful, which they are not). Ordinal regression models the ORDER directly:
one direction in feature space moves you monotonically up the scale.
"""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, check_is_fitted
from ..utils import check_X_y, check_array
from ..preprocessing import LabelEncoder


class OrdinalLogistic(BaseEstimator, ClassifierMixin):
    """Proportional-odds (cumulative logit) model -- the classic ordinal method.

    THE MODEL
    ---------
    A single weight vector ``w`` projects each point onto a latent scale, and
    ordered thresholds ``b_0 < b_1 < ... < b_{K-2}`` cut that scale into the K
    ordered categories::

        P(y <= k | x) = sigmoid(b_k - x·w)

    So one score ``x·w`` decides the category, and moving it up crosses the
    thresholds in order. "Proportional odds" because the SAME ``w`` governs every
    threshold -- the effect of a feature on the odds of being above any cutoff is
    the same, which is the assumption that makes the model parsimonious and
    interpretable (one coefficient per feature, not one per class).

    FIT
    ---
    Maximum likelihood. The thresholds are kept ordered by parameterising them as
    a base plus non-negative softplus increments, so the optimiser cannot cross
    them. Optimised with L-BFGS.

    McCullagh (1980).
    """

    def __init__(self, max_iter=200):
        self.max_iter = max_iter

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _thresholds(self, params):
        b0 = params[0]
        deltas = np.log1p(np.exp(params[1:self.K_ - 1]))   # softplus >= 0
        return np.concatenate([[b0], b0 + np.cumsum(deltas)])

    def _probs(self, X, w, thr):
        z = X @ w
        cum = self._sigmoid(thr[None, :] - z[:, None])     # P(y<=k), k=0..K-2
        left = np.hstack([np.zeros((len(X), 1)), cum])
        right = np.hstack([cum, np.ones((len(X), 1))])
        return np.clip(right - left, 1e-12, 1.0)           # P(y==k)

    def fit(self, X, y):
        from scipy.optimize import minimize
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        yi = self._le.transform(y)
        self.K_ = len(self.classes_)
        d = X.shape[1]

        def nll(params):
            w = params[:d]
            thr = self._thresholds(params[d:])
            p = self._probs(X, w, thr)
            return -np.mean(np.log(p[np.arange(len(X)), yi]))

        init = np.concatenate([np.zeros(d), [0.0], np.zeros(self.K_ - 2)])
        res = minimize(nll, init, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        self.coef_ = res.x[:d]
        self.thresholds_ = self._thresholds(res.x[d:])
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "coef_")
        return self._probs(check_array(X), self.coef_, self.thresholds_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class OrdinalRidge(BaseEstimator, ClassifierMixin):
    """A pragmatic ordinal baseline: regress on the ranks, then snap to a class.

    Encode the ordered classes as 0..K-1, fit a plain ridge regression to those
    integers (which, unlike a classifier, is rewarded for landing CLOSE -- so a
    "good" predicted as "excellent" costs less than as "poor"), then round the
    continuous prediction to the nearest class. Cruder than the proportional-odds
    model -- it does assume equal spacing -- but fast, and often a strong baseline.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        from ..linear_model import Ridge
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self.reg_ = Ridge(alpha=self.alpha).fit(X, self._le.transform(y).astype(float))
        return self

    def predict(self, X):
        check_is_fitted(self, "reg_")
        raw = np.rint(self.reg_.predict(check_array(X)))
        idx = np.clip(raw, 0, len(self.classes_) - 1).astype(int)
        return self.classes_[idx]


__all__ = ["OrdinalLogistic", "OrdinalRidge"]
