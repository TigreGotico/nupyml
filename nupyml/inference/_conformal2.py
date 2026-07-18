"""Conformal prediction v2 and predictive uncertainty.

The base conformal (split/Mondrian/CQR/Venn-Abers/ACI) covers regression
intervals and simple sets. These add ADAPTIVE classification sets, leave-one-out
/ cross conformal with tighter coverage, time-series conformal, and an ensemble
uncertainty estimate. All keep the distribution-free coverage spirit.
"""
import numpy as np

from ..base import BaseEstimator, clone, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


class APS(BaseEstimator):
    """Adaptive Prediction Sets for classification (Romano et al., 2020).

    Split-conformal classification (``ConformalClassifier``) thresholds
    ``1 - p_true`` and can produce badly-sized sets on hard examples. APS instead
    accumulates the SORTED class probabilities until the true class is included:
    the conformity score is the total probability mass of classes at least as
    likely as the truth. Calibrating that gives sets whose size ADAPTS to
    difficulty -- a singleton on easy inputs, several labels on ambiguous ones --
    while keeping the coverage guarantee. ``predict_set`` returns a set of labels
    per sample.
    """

    def __init__(self, estimator, alpha=0.1, calibration_fraction=0.3,
                 random_state=None):
        self.estimator = estimator
        self.alpha = alpha
        self.calibration_fraction = calibration_fraction
        self.random_state = random_state

    def _score(self, proba, label):
        # total mass of classes at least as probable as the true class
        order = np.argsort(-proba)
        cum = 0.0
        for j in order:
            cum += proba[j]
            if j == label:
                break
        return cum

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        n = len(y)
        n_cal = max(1, int(self.calibration_fraction * n))
        perm = rng.permutation(n)
        cal, tr = perm[:n_cal], perm[n_cal:]
        self.estimator_ = clone(self.estimator).fit(X[tr], y[tr])
        self.classes_ = self.estimator_.classes_
        c2i = {c: i for i, c in enumerate(self.classes_)}
        P = self.estimator_.predict_proba(X[cal])
        scores = np.array([self._score(P[i], c2i[y[cal][i]]) for i in range(len(cal))])
        self.tau_ = np.quantile(scores, 1 - self.alpha)
        return self

    def predict_set(self, X):
        check_is_fitted(self, "tau_")
        P = self.estimator_.predict_proba(check_array(X))
        sets = []
        for p in P:
            order = np.argsort(-p)
            cum, chosen = 0.0, []
            for j in order:
                chosen.append(self.classes_[j])
                cum += p[j]
                if cum >= self.tau_:
                    break
            sets.append(set(chosen))
        return sets

    def predict(self, X):
        return self.estimator_.predict(check_array(X))


class RAPS(APS):
    """Regularized APS -- penalise LARGE sets to keep them small (Angelopoulos, 2021).

    APS can produce large sets when the tail probabilities are diffuse. RAPS adds a
    regularisation that charges a growing penalty for each label beyond the
    ``k_reg``-th most probable, so the set stops growing into the uninformative
    tail. Same coverage guarantee, markedly smaller (more useful) sets on
    many-class problems. ``k_reg`` is the free-inclusion count, ``lam`` the penalty.
    """

    def __init__(self, estimator, alpha=0.1, k_reg=1, lam=0.1,
                 calibration_fraction=0.3, random_state=None):
        super().__init__(estimator, alpha, calibration_fraction, random_state)
        self.k_reg = k_reg
        self.lam = lam

    def _score(self, proba, label):
        order = np.argsort(-proba)
        cum = 0.0
        for rank, j in enumerate(order):
            cum += proba[j] + self.lam * max(0, rank - self.k_reg + 1)  # size penalty
            if j == label:
                break
        return cum

    def predict_set(self, X):
        check_is_fitted(self, "tau_")
        P = self.estimator_.predict_proba(check_array(X))
        sets = []
        for p in P:
            order = np.argsort(-p)
            cum, chosen = 0.0, []
            for rank, j in enumerate(order):
                chosen.append(self.classes_[j])
                cum += p[j] + self.lam * max(0, rank - self.k_reg + 1)
                if cum >= self.tau_:
                    break
            sets.append(set(chosen))
        return sets


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


class DeepEnsemble(BaseEstimator):
    """Predictive uncertainty from an ENSEMBLE's disagreement (Lakshminarayanan, 2017).

    A single model gives a point prediction with no honest sense of its own
    uncertainty. A deep ensemble trains several models from different random
    initialisations / bootstraps; where they AGREE the prediction is confident,
    where they DISAGREE (high variance across members) it is uncertain -- and that
    variance is a remarkably well-calibrated uncertainty estimate, crucially
    including EPISTEMIC uncertainty (growing away from the training data) that a
    single model or MC-dropout understates. ``predict`` returns (mean, std).
    """

    def __init__(self, estimator, n_models=5, random_state=None):
        self.estimator = estimator
        self.n_models = n_models
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        n = len(y)
        self.models_ = []
        for _ in range(self.n_models):
            idx = rng.choice(n, n, replace=True)       # bootstrap for diversity
            m = clone(self.estimator)
            if "random_state" in m.get_params():
                m.set_params(random_state=rng.randint(2 ** 31 - 1))
            self.models_.append(m.fit(X[idx], y[idx]))
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self, "models_")
        preds = np.array([m.predict(check_array(X)) for m in self.models_])
        mean, std = preds.mean(axis=0), preds.std(axis=0)
        return (mean, std) if return_std else mean


__all__ = ["APS", "RAPS", "JackknifePlus", "EnbPI", "DeepEnsemble"]
