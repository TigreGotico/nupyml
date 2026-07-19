"""Ensemble batch prediction intervals for TIME SERIES (Xu & Xie, 2021)."""
import numpy as np
from ..base import BaseEstimator, clone, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


class EnbPI(BaseEstimator):
    """Ensemble batch prediction intervals for TIME SERIES (Xu & Xie, 2021).

    Split conformal assumes exchangeability, which time series violate. EnbPI fits
    a BOOTSTRAP ensemble and, for each training point, forms an OUT-OF-BAG
    prediction (from the trees that did not see it) -- so every residual is
    effectively out-of-sample without a held-out split. The interval width is a
    quantile of those OOB residuals, and it can be refreshed online as new errors
    arrive. Distribution-free intervals that hold under the temporal dependence
    ordinary conformal breaks on.
    """

    def __init__(self, estimator, alpha=0.1, n_estimators=20, random_state=None):
        self.estimator = estimator
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        n = len(y)
        self.models_ = []
        oob_preds = np.full((self.n_estimators, n), np.nan)
        for b in range(self.n_estimators):
            idx = rng.choice(n, n, replace=True)
            m = clone(self.estimator).fit(X[idx], y[idx])
            self.models_.append(m)
            oob = np.setdiff1d(np.arange(n), idx)      # out-of-bag samples
            if len(oob):
                oob_preds[b, oob] = m.predict(X[oob])
        agg = np.nanmean(oob_preds, axis=0)            # OOB ensemble prediction
        valid = ~np.isnan(agg)
        self.residuals_ = np.abs(y[valid] - agg[valid])
        return self

    def predict_interval(self, X, coverage=None):
        check_is_fitted(self, "models_")
        X = check_array(X)
        alpha = self.alpha if coverage is None else 1 - coverage
        pred = np.mean([m.predict(X) for m in self.models_], axis=0)
        q = np.quantile(self.residuals_, 1 - alpha)
        return pred - q, pred + q

    def predict(self, X):
        return np.mean([m.predict(check_array(X)) for m in self.models_], axis=0)


__all__ = ["EnbPI"]
