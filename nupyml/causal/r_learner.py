"""Isolate the heterogeneous effect by RESIDUALISING (Nie & Wager, 2021)."""
import numpy as np
from ..base import BaseEstimator, clone
from ..utils import check_array, check_random_state
from ..tree import DecisionTreeRegressor


class RLearner(BaseEstimator):
    """Isolate the heterogeneous effect by RESIDUALISING (Nie & Wager, 2021).

    Confounding contaminates a naive effect estimate: outcome and treatment both
    depend on covariates. The R-learner removes that in two stages. First fit the
    "nuisances" -- ``m(x) = E[Y|X]`` and the propensity ``e(x) = E[T|X]`` -- and take
    RESIDUALS ``Y - m(x)`` and ``T - e(x)``. Robinson's transformation shows the
    heterogeneous effect ``tau(x)`` is exactly the regression of the outcome residual
    on the treatment residual, so a weighted least-squares on those residuals
    recovers ``tau`` free of the confounding both stages absorbed. Linear ``tau(x)``
    here.
    """

    def __init__(self, outcome_model=None, propensity_model=None):
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model

    def fit(self, X, treatment, y):
        X = check_array(X)
        t = np.asarray(treatment, float); y = np.asarray(y, float)
        om = clone(self.outcome_model) if self.outcome_model is not None \
            else DecisionTreeRegressor(max_depth=4)
        pm = clone(self.propensity_model) if self.propensity_model is not None \
            else DecisionTreeRegressor(max_depth=4)
        m_hat = om.fit(X, y).predict(X)                   # E[Y|X]
        e_hat = np.clip(pm.fit(X, t).predict(X), 0.02, 0.98)   # E[T|X]
        y_res = y - m_hat
        t_res = t - e_hat
        # Robinson: minimise sum (y_res - t_res * X @ theta)^2  => weighted LS
        Xd = np.column_stack([np.ones(len(y)), X]) * t_res[:, None]
        theta, *_ = np.linalg.lstsq(Xd, y_res, rcond=None)
        self.theta_ = theta
        self._m, self._e = om, pm
        return self

    def predict(self, X):
        X = check_array(X)
        return np.column_stack([np.ones(len(X)), X]) @ self.theta_


__all__ = ["RLearner"]
