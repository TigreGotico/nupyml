"""A heavy-tailed alternative to the Gaussian process (Shah et al., 2014)."""
import numpy as np
from .kernels import Kernel, RBF
from ..base import BaseEstimator, RegressorMixin, check_is_fitted
from ..utils import check_X_y, check_array


class StudentTProcess(BaseEstimator, RegressorMixin):
    """A heavy-tailed alternative to the Gaussian process (Shah et al., 2014).

    A GP's predictive distribution is Gaussian, so it is confident that large
    deviations are essentially impossible -- brittle when the data has outliers or
    heavier-than-Gaussian noise. The Student-t process replaces the Gaussian with a
    multivariate STUDENT-T, whose degrees-of-freedom ``nu`` control the tail
    weight (``nu -> inf`` recovers the GP). Its predictive VARIANCE also scales with
    how well the observed data fit the prior (``beta = y' K^{-1} y``), so it
    widens the intervals AFTER seeing surprising data -- a data-dependent
    uncertainty a GP lacks. Same closed-form mean as a GP; a t-scaled variance.
    """

    def __init__(self, kernel=None, alpha=1e-6, nu=5.0):
        self.kernel = kernel
        self.alpha = alpha
        self.nu = nu

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.kernel_ = self.kernel or RBF()
        self.X_ = X
        self.y_mean_ = y.mean()
        self.y_ = y - self.y_mean_
        K = self.kernel_(X, X) + self.alpha * np.eye(len(X))
        self.K_inv_ = np.linalg.inv(K)
        self.beta_ = float(self.y_ @ self.K_inv_ @ self.y_)   # data-fit scale
        self.n_ = len(X)
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self, "K_inv_")
        X = check_array(X)
        Ks = self.kernel_(X, self.X_)
        mean = Ks @ self.K_inv_ @ self.y_ + self.y_mean_
        if not return_std:
            return mean
        var = self.kernel_.diag(X) - np.einsum("ij,jk,ik->i", Ks, self.K_inv_, Ks)
        # Student-t predictive scale: inflate by (nu + beta - 2)/(nu + n - 2)
        scale = (self.nu + self.beta_ - 2) / (self.nu + self.n_ - 2)
        return mean, np.sqrt(np.maximum(var, 0) * scale)


__all__ = ["StudentTProcess"]
