"""Two-stage least squares -- identify an effect despite UNOBSERVED confounding."""
import numpy as np
from ..base import BaseEstimator, clone
from ..utils import check_array


class InstrumentalVariables(BaseEstimator):
    """Two-stage least squares -- identify an effect despite UNOBSERVED confounding.

    THE PROBLEM ADJUSTMENT CANNOT FIX
    ---------------------------------
    IPW and regression only remove confounding you can MEASURE. If an unobserved
    variable drives both the treatment and the outcome (ability drives both
    schooling and wages), no amount of adjusting for observed covariates recovers
    the causal effect. An INSTRUMENT ``Z`` breaks the deadlock: it moves the
    treatment but affects the outcome ONLY through the treatment (distance to
    college shifts schooling but not wages directly).

    2SLS uses it in two regressions: (1) regress treatment on the instrument to
    get the part of treatment the instrument EXPLAINS (confounder-free by
    assumption), then (2) regress the outcome on THAT predicted treatment. The
    second stage's slope is the causal effect. ``fit(Z, T, y)``; the effect is in
    ``coef_``.
    """

    def fit(self, Z, T, y):
        Z = check_array(Z)
        T = np.asarray(T, float)
        y = np.asarray(y, float)
        Z1 = np.column_stack([np.ones(len(Z)), Z])
        # stage 1: project treatment onto the instrument(s)
        beta1, *_ = np.linalg.lstsq(Z1, T, rcond=None)
        T_hat = Z1 @ beta1
        # stage 2: regress outcome on the fitted (confounder-free) treatment
        D = np.column_stack([np.ones(len(y)), T_hat])
        beta2, *_ = np.linalg.lstsq(D, y, rcond=None)
        self.intercept_, self.coef_ = beta2[0], beta2[1]
        return self


__all__ = ["InstrumentalVariables"]
