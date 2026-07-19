"""CATE by fitting SEPARATE outcome models per treatment arm."""
import numpy as np
from ..base import BaseEstimator, clone
from ..utils import check_array


class TLearner(BaseEstimator):
    """CATE by fitting SEPARATE outcome models per treatment arm.

    Heterogeneous treatment effects ask how the effect VARIES across individuals.
    The T-learner fits one regressor on the treated, another on the controls, and
    estimates each unit's effect as the difference of the two predictions::

        tau(x) = mu_treated(x) - mu_control(x)

    Simple and flexible, but it splits the data and can't share strength between
    arms -- the X-learner refines exactly that. ``predict_effect(X)`` returns the
    per-unit CATE.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, T, y):
        from ..ensemble import RandomForestRegressor
        X = check_array(X); T = np.asarray(T); y = np.asarray(y, float)
        base = self.base_estimator or RandomForestRegressor(n_estimators=100,
                                                            random_state=0)
        self.m1_ = clone(base).fit(X[T == 1], y[T == 1])
        self.m0_ = clone(base).fit(X[T == 0], y[T == 0])
        return self

    def predict_effect(self, X):
        X = check_array(X)
        return self.m1_.predict(X) - self.m0_.predict(X)


__all__ = ["TLearner"]
