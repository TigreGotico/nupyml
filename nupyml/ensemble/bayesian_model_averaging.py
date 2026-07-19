"""Weight models by their POSTERIOR probability, not pick one (Hoeting, 1999)."""
import numpy as np
from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin, clone)
from ..utils import check_array, check_random_state


class BayesianModelAveraging(BaseEstimator, RegressorMixin):
    """Weight models by their POSTERIOR probability, not pick one (Hoeting, 1999).

    Choosing a single "best" model ignores that others were nearly as plausible and
    understates uncertainty. Bayesian model averaging keeps them all and weights
    each by its posterior probability, approximated here from the BIC (which trades
    fit against complexity): ``w_m ∝ exp(-BIC_m / 2)``. The prediction is the
    weighted average, so a model the data barely prefers still contributes, and
    predictive uncertainty reflects model uncertainty rather than pretending the
    chosen model is certainly correct.
    """

    def __init__(self, estimators, n_params=None):
        self.estimators = estimators
        self.n_params = n_params

    def fit(self, X, y):
        X = check_array(X); y = np.asarray(y, float)
        n = len(y)
        bics = []
        self.models_ = []
        for i, e in enumerate(self.estimators):
            m = clone(e).fit(X, y)
            self.models_.append(m)
            rss = np.sum((y - m.predict(X)) ** 2)
            k = (self.n_params[i] if self.n_params else X.shape[1] + 1)
            bic = n * np.log(rss / n + 1e-12) + k * np.log(n)   # ~ -2 log-evidence
            bics.append(bic)
        bics = np.array(bics)
        rel = np.exp(-0.5 * (bics - bics.min()))         # posterior model weights
        self.weights_ = rel / rel.sum()
        return self

    def predict(self, X):
        X = check_array(X)
        P = np.column_stack([m.predict(X) for m in self.models_])
        return P @ self.weights_


__all__ = ["BayesianModelAveraging"]
