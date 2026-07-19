"""Predictive uncertainty from an ENSEMBLE's disagreement (Lakshminarayanan, 2017)."""
import numpy as np
from ..base import BaseEstimator, clone, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


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


__all__ = ["DeepEnsemble"]
