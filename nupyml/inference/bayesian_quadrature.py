"""Bayesian quadrature: numerical integration as GP regression, with error bars."""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array


class BayesianQuadrature(BaseEstimator):
    """Integration as GP regression, WITH error bars (O'Hagan, 1991).

    Monte-Carlo integration converges as ``1/sqrt(n)`` and gives only a noisy point
    estimate. Bayesian quadrature instead puts a Gaussian-process prior on the
    integrand, conditions it on the function evaluations, and integrates the GP
    posterior -- which has a CLOSED FORM for an RBF kernel. The result is not just an
    estimate of the integral but a full posterior over it (a mean and a variance), so
    you know how much to trust it and where to sample next. Far more sample-efficient
    than Monte Carlo for smooth functions. Uniform measure on ``[a, b]`` here.
    """

    def __init__(self, length_scale=0.3, noise=1e-8):
        self.length_scale = length_scale
        self.noise = noise

    def _kernel(self, A, B):
        from scipy.spatial.distance import cdist
        return np.exp(-0.5 * cdist(A, B, "sqeuclidean") / self.length_scale ** 2)

    def fit(self, X, y, a=0.0, b=1.0):
        X = check_array(X); y = np.asarray(y, float).ravel()
        self.X_ = X; self.a, self.b = a, b
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        self.Kinv_ = np.linalg.inv(K)
        # kernel mean z_i = ∫ k(x, x_i) dx over [a, b] (RBF integrates to erf terms)
        from scipy.special import erf
        l = self.length_scale
        z = (np.sqrt(np.pi / 2) * l
             * (erf((b - X[:, 0]) / (np.sqrt(2) * l))
                - erf((a - X[:, 0]) / (np.sqrt(2) * l))))
        self.z_ = z
        self.integral_ = z @ self.Kinv_ @ y
        # posterior variance of the integral
        zz = np.sqrt(np.pi) * l ** 2 * (
            (b - a) / l * erf((b - a) / (2 * l))
            + 2 / np.sqrt(np.pi) * (np.exp(-((b - a) ** 2) / (4 * l ** 2)) - 1))
        self.variance_ = max(zz - z @ self.Kinv_ @ z, 0.0)
        return self

    def integral(self, return_std=False):
        if return_std:
            return self.integral_, np.sqrt(self.variance_)
        return self.integral_


__all__ = ["BayesianQuadrature"]
