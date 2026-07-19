"""Regularized APS -- penalise LARGE sets to keep them small (Angelopoulos, 2021)."""
import numpy as np
from ..base import BaseEstimator, clone, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state
from .aps import APS


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


__all__ = ["RAPS"]
