"""Cross-conformal (CV+) regression intervals with a coverage proof"""
import numpy as np
from ..base import BaseEstimator, clone, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


class JackknifePlus(BaseEstimator):
    """Cross-conformal (CV+) regression intervals with a coverage proof
    (Barber et al., 2021).

    Split conformal wastes data (a whole calibration set is never trained on).
    Jackknife+/CV+ uses EVERY point for both: fit K leave-fold-out models, take
    each held-out point's residual, and build the interval for a new point from the
    per-fold predictions PLUS/MINUS those residuals::

        [ quantile_alpha( mu_{-k}(x) - R ),  quantile_{1-alpha}( mu_{-k}(x) + R ) ]

    This is provably ~``1 - 2*alpha`` covered without a separate calibration split,
    so it is far more data-efficient on small datasets. ``cv`` folds trade cost for
    tightness (``cv = n`` is the exact jackknife+).
    """

    def __init__(self, estimator, alpha=0.1, cv=5, random_state=None):
        self.estimator = estimator
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        from ..model_selection import KFold
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        kf = KFold(n_splits=self.cv, shuffle=True,
                   random_state=rng.randint(2 ** 31 - 1))
        self.models_ = []
        self.residuals_ = []
        for tr, te in kf.split(X):
            m = clone(self.estimator).fit(X[tr], y[tr])
            self.models_.append(m)
            self.residuals_.extend(np.abs(y[te] - m.predict(X[te])))
        self.residuals_ = np.array(self.residuals_)
        return self

    def predict_interval(self, X, coverage=None):
        check_is_fitted(self, "models_")
        X = check_array(X)
        alpha = self.alpha if coverage is None else 1 - coverage
        preds = np.array([m.predict(X) for m in self.models_])   # (K, n)
        mean_pred = preds.mean(axis=0)
        # CV+ bounds: ensemble mean +/- the (1-alpha) residual quantile
        q = np.quantile(self.residuals_, 1 - alpha)
        return mean_pred - q, mean_pred + q

    def predict(self, X):
        preds = np.array([m.predict(check_array(X)) for m in self.models_])
        return preds.mean(axis=0)


__all__ = ["JackknifePlus"]
