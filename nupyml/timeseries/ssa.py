"""Singular Spectrum Analysis: decompose a series by SVD of its trajectory"""
import numpy as np
from ..base import BaseEstimator


class SSA(BaseEstimator):
    """Singular Spectrum Analysis: decompose a series by SVD of its trajectory
    matrix.

    SSA is model-free spectral decomposition. Slide a window over the series to
    build a TRAJECTORY MATRIX (each column a lagged snapshot); its SVD yields
    components that, grouped, separate TREND, oscillations, and noise -- without
    assuming a parametric model. Reconstruct a component by diagonal-averaging its
    rank-1 piece back into a series. Widely used to extract trend/seasonality and
    to denoise. ``n_components`` sets how many leading components to keep.
    """

    def __init__(self, window=None, n_components=2):
        self.window = window
        self.n_components = n_components

    def fit(self, y):
        y = np.asarray(y, float)
        n = len(y)
        L = self.window or n // 3
        K = n - L + 1
        X = np.array([y[i:i + L] for i in range(K)]).T   # trajectory matrix (L x K)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        self.components_ = []
        for k in range(min(self.n_components, len(s))):
            Xk = s[k] * np.outer(U[:, k], Vt[k])
            self.components_.append(self._diag_avg(Xk, n))
        self.reconstruction_ = np.sum(self.components_, axis=0)
        return self

    @staticmethod
    def _diag_avg(X, n):
        """Hankelise: average each anti-diagonal back into a 1-D series."""
        L, K = X.shape
        out = np.zeros(n)
        cnt = np.zeros(n)
        for i in range(L):
            for j in range(K):
                out[i + j] += X[i, j]; cnt[i + j] += 1
        return out / cnt


__all__ = ["SSA"]
