"""CATE that IMPUTES individual effects then models them (Künzel et al., 2019)."""
import numpy as np
from ..base import BaseEstimator, clone
from ..utils import check_array


class XLearner(BaseEstimator):
    """CATE that IMPUTES individual effects then models them (Künzel et al., 2019).

    The T-learner wastes information when the arms are imbalanced. The X-learner
    adds a clever middle step: use each arm's model to IMPUTE the counterfactual
    for the OTHER arm, giving a pseudo-effect per unit, then fit a model to those
    pseudo-effects in each arm and blend the two by the propensity score. This
    borrows strength across arms and shines when one group is much larger -- the
    common case in observational data. ``predict_effect(X)`` returns the CATE.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, T, y):
        from ..ensemble import RandomForestRegressor
        from ..linear_model import LogisticRegression
        X = check_array(X); T = np.asarray(T); y = np.asarray(y, float)
        base = self.base_estimator or RandomForestRegressor(n_estimators=100,
                                                            random_state=0)
        m1 = clone(base).fit(X[T == 1], y[T == 1])
        m0 = clone(base).fit(X[T == 0], y[T == 0])
        # imputed treatment effects within each arm
        d1 = y[T == 1] - m0.predict(X[T == 1])
        d0 = m1.predict(X[T == 0]) - y[T == 0]
        self.tau1_ = clone(base).fit(X[T == 1], d1)
        self.tau0_ = clone(base).fit(X[T == 0], d0)
        self.prop_ = LogisticRegression(max_iter=300).fit(X, T)
        return self

    def predict_effect(self, X):
        X = check_array(X)
        g = self.prop_.predict_proba(X)[:, 1]         # propensity as the blend weight
        return g * self.tau0_.predict(X) + (1 - g) * self.tau1_.predict(X)


__all__ = ["XLearner"]
