"""Compress many series into a few COMMON factors (Geweke, 1977)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class DynamicFactorModel(BaseEstimator):
    """Compress many series into a few COMMON factors (Geweke, 1977).

    Dozens of macro indicators, or hundreds of sensors, are mostly driven by a
    handful of shared forces -- a business cycle, a temperature field. A dynamic
    factor model extracts a few latent FACTORS (here via principal components of the
    standardised series) that explain their comovement, models the factors' own
    dynamics with a VAR, and reconstructs every series as a loading on those factors
    plus idiosyncratic noise. Forecasting the few factors and mapping back forecasts
    all series at once, with far fewer parameters than a full VAR. ``n_factors`` is
    the number of common drivers.
    """

    def __init__(self, n_factors=2, var_order=1):
        self.n_factors = n_factors
        self.var_order = var_order

    def fit(self, Y):
        Y = check_array(Y)
        self.mean_ = Y.mean(axis=0); self.std_ = Y.std(axis=0) + 1e-8
        Z = (Y - self.mean_) / self.std_
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        self.loadings_ = Vt[:self.n_factors]               # (r, k)
        F = Z @ self.loadings_.T                            # factors (T, r)
        self.factors_ = F
        # VAR(1) on the factors
        rows = np.column_stack([np.ones(len(F) - 1), F[:-1]])
        self.var_coef_, *_ = np.linalg.lstsq(rows, F[1:], rcond=None)
        return self

    def forecast(self, steps=10):
        f = self.factors_[-1].copy()
        out = []
        for _ in range(steps):
            f = np.concatenate([[1.0], f]) @ self.var_coef_
            z = f @ self.loadings_                          # reconstruct series
            out.append(z * self.std_ + self.mean_)
        return np.array(out)


__all__ = ["DynamicFactorModel"]
