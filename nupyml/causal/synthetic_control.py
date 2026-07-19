"""Build a weighted combination of controls that mimics the treated unit"""
import numpy as np
from ..base import BaseEstimator, clone
from ..utils import check_array


class SyntheticControl(BaseEstimator):
    """Build a weighted combination of controls that mimics the treated unit
    (Abadie & Gardeazabal, 2003).

    When there is ONE treated unit (a country, a state) and several controls, no
    single control is a good counterfactual. Synthetic control finds convex
    weights over the controls so their weighted PRE-treatment outcomes track the
    treated unit's; that "synthetic" unit's POST-treatment path is the
    counterfactual, and the gap from the real treated path is the effect. The
    weights are non-negative and sum to one, so the synthetic unit stays an
    interpolation of real units (no extrapolation). ``fit`` on pre-period matrices,
    ``effect`` on the post period.
    """

    def __init__(self, max_iter=5000, lr=0.01):
        self.max_iter = max_iter
        self.lr = lr

    def fit(self, treated_pre, controls_pre):
        # controls_pre: (T_pre, n_controls); treated_pre: (T_pre,)
        Y = check_array(controls_pre)
        t = np.asarray(treated_pre, float)
        n = Y.shape[1]
        w = np.full(n, 1.0 / n)
        for _ in range(self.max_iter):                # projected gradient on the simplex
            grad = Y.T @ (Y @ w - t)
            w = w - self.lr * grad / len(t)
            w = np.maximum(w, 0)
            s = w.sum()
            w = w / s if s > 0 else np.full(n, 1.0 / n)
        self.weights_ = w
        return self

    def effect(self, treated_post, controls_post):
        synthetic = check_array(controls_post) @ self.weights_
        return np.asarray(treated_post, float) - synthetic


__all__ = ["SyntheticControl"]
