"""Double / debiased machine learning for the average treatment effect"""
import numpy as np
from ..base import BaseEstimator, clone
from ..utils import check_array


class DoubleML(BaseEstimator):
    """Double / debiased machine learning for the average treatment effect
    (Chernozhukov et al., 2018).

    THE ORTHOGONALISATION
    ---------------------
    You want to control for confounders with flexible ML models, but plugging ML
    predictions straight in biases the effect (regularisation bleaks into the
    estimate). Double ML removes this: fit two nuisance models -- outcome ``E[y|X]``
    and treatment ``E[T|X]`` -- then regress the RESIDUALS of ``y`` on the
    RESIDUALS of ``T``. Partialling both sides out (Frisch-Waugh-Lovell) makes the
    estimate ORTHOGONAL to small nuisance errors, so ML-grade confounder models are
    safe to use. CROSS-FITTING (predict each fold from the others) removes the
    remaining overfitting bias. ``effect_`` is the estimated ATE.
    """

    def __init__(self, outcome_model=None, treatment_model=None, cv=2,
                 random_state=None):
        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, T, y):
        from ..ensemble import RandomForestRegressor
        from ..model_selection import KFold
        X = check_array(X)
        T = np.asarray(T, float)
        y = np.asarray(y, float)
        om = self.outcome_model or RandomForestRegressor(n_estimators=50,
                                                         random_state=0)
        tm = self.treatment_model or RandomForestRegressor(n_estimators=50,
                                                          random_state=0)
        y_res = np.zeros_like(y)
        t_res = np.zeros_like(T)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        for train, test in kf.split(X):
            # cross-fitting: nuisance predictions from the OTHER fold
            y_res[test] = y[test] - clone(om).fit(X[train], y[train]).predict(X[test])
            t_res[test] = T[test] - clone(tm).fit(X[train], T[train]).predict(X[test])
        # effect = slope of residual-y on residual-T
        self.effect_ = float(np.sum(t_res * y_res) / (np.sum(t_res ** 2) + 1e-12))
        return self


__all__ = ["DoubleML"]
