"""Gaussian processes: a distribution over functions.

THE IDEA
--------
Rather than fitting parameters of one function, put a prior over ALL functions
and condition it on the data. A GP says: any finite set of function values is
jointly Gaussian, with covariance given by a kernel::

    cov(f(x), f(x')) = k(x, x')

The kernel is the entire model. It says what "similar inputs" means, and
therefore what kind of function this is -- ``RBF`` for smooth, ``Matern`` for
rougher, ``ExpSineSquared`` for periodic. Choosing the kernel replaces choosing
an architecture, and kernels compose with ``+`` and ``*``, so "smooth trend plus
periodic seasonality plus noise" is written literally as a sum.

WHAT YOU GET FOR IT
-------------------
Predictions come with honest UNCERTAINTY, not as an add-on but from the same
Gaussian conditioning that produces the mean. And the uncertainty behaves
correctly: near the data it is small, far away it grows back to the prior. That
is what makes GPs the tool for Bayesian optimization -- you can ask where the
model does not know.

WHAT IT COSTS
-------------
Conditioning needs the inverse of an n-by-n covariance: O(n^3) time and O(n^2)
memory. Perfectly comfortable at a thousand points, hopeless at a million. GPs
are for small data where each observation is expensive -- which is exactly when
uncertainty matters most.

Hyperparameters (length scales, noise) are chosen by maximising the MARGINAL
likelihood, which automatically balances fit against complexity -- an Occam's
razor that falls out of the mathematics rather than being imposed.

Implementation-wise: nothing is ever inverted. Cholesky factorisation gives both
the solve and the log-determinant stably and in half the arithmetic.
``GaussianProcessClassifier`` needs one more approximation, since a Gaussian
prior through a sigmoid link has no closed-form posterior -- it uses Laplace's
method, fitting a Gaussian at the posterior mode.
"""
import numpy as np
import scipy.linalg
import scipy.optimize
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, sigmoid


from .kernels import (Kernel, ConstantKernel, WhiteKernel, RBF, Matern,
                      RationalQuadratic, ExpSineSquared, DotProduct, Sum,
                      Product)


def ConstantTimes(kernel, scale=1.0):
    """Backwards-compatible helper: scale**2 * kernel."""
    return ConstantKernel(scale ** 2) * kernel


class GaussianProcessRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, kernel=None, alpha=1e-10, optimize=True, n_restarts=0,
                 random_state=None):
        self.kernel = kernel
        self.alpha = alpha
        self.optimize = optimize
        self.n_restarts = n_restarts
        self.random_state = random_state

    def _nll(self, theta, X, y, kernel):
        kernel.theta = theta
        K = kernel(X, X) + self.alpha * np.eye(len(X))
        try:
            L = scipy.linalg.cholesky(K, lower=True)
        except scipy.linalg.LinAlgError:
            return 1e25
        alpha_vec = scipy.linalg.cho_solve((L, True), y)
        return float(0.5 * y @ alpha_vec + np.log(np.diag(L)).sum()
                     + 0.5 * len(X) * np.log(2 * np.pi))

    def fit(self, X, y):
        from ..utils import check_random_state
        X, y = check_X_y(X, y, y_numeric=True)
        kernel = (self.kernel or ConstantKernel(1.0) * RBF(1.0)).clone()
        self._y_mean = y.mean()
        yc = y - self._y_mean
        if self.optimize:
            rng = check_random_state(self.random_state)
            best = (np.inf, kernel.theta)
            starts = [kernel.theta] + [rng.uniform(-2, 2, size=len(kernel.theta))
                                       for _ in range(self.n_restarts)]
            for t0 in starts:
                res = scipy.optimize.minimize(
                    self._nll, t0, args=(X, yc, kernel), method="L-BFGS-B")
                if res.fun < best[0]:
                    best = (res.fun, res.x)
            kernel.theta = best[1]
            self.log_marginal_likelihood_ = -best[0]
        self.kernel_ = kernel
        K = kernel(X, X) + self.alpha * np.eye(len(X))
        self._L = scipy.linalg.cholesky(K, lower=True)
        self._alpha_vec = scipy.linalg.cho_solve((self._L, True), yc)
        self._X_train = X
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self, "kernel_")
        X = check_array(X)
        Ks = self.kernel_(X, self._X_train)
        mean = Ks @ self._alpha_vec + self._y_mean
        if not return_std:
            return mean
        v = scipy.linalg.solve_triangular(self._L, Ks.T, lower=True)
        var = np.maximum(self.kernel_.diag(X) - (v ** 2).sum(axis=0), 0)
        return mean, np.sqrt(var)


class GaussianProcessClassifier(BaseEstimator, ClassifierMixin):
    """Binary GP classification via Laplace approximation (RBF kernel)."""

    _estimator_tags = {"binary_only": True}

    def __init__(self, kernel=None, max_iter=100):
        self.kernel = kernel
        self.max_iter = max_iter

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        if len(self.classes_) != 2:
            raise ValueError("GaussianProcessClassifier is binary only")
        t = self._le.transform(y).astype(np.float64)  # {0,1}
        kernel = (self.kernel or ConstantKernel(1.0) * RBF(1.0)).clone()
        K = kernel(X, X) + 1e-8 * np.eye(len(X))
        f = np.zeros(len(X))
        # Newton iterations for the Laplace mode
        for _ in range(self.max_iter):
            pi = sigmoid(f)
            W = pi * (1 - pi)
            sqrtW = np.sqrt(W)
            B = np.eye(len(X)) + sqrtW[:, None] * K * sqrtW[None, :]
            L = scipy.linalg.cholesky(B, lower=True)
            b = W * f + (t - pi)
            a = b - sqrtW * scipy.linalg.cho_solve(
                (L, True), sqrtW * (K @ b))
            f_new = K @ a
            if np.max(np.abs(f_new - f)) < 1e-8:
                f = f_new
                break
            f = f_new
        self._X_train = X
        self._kernel = kernel
        pi = sigmoid(f)
        self._grad = t - pi
        W = pi * (1 - pi)
        sqrtW = np.sqrt(W)
        self._sqrtW = sqrtW
        B = np.eye(len(X)) + sqrtW[:, None] * K * sqrtW[None, :]
        self._L = scipy.linalg.cholesky(B, lower=True)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_X_train")
        X = check_array(X)
        Ks = self._kernel(X, self._X_train)
        f_mean = Ks @ self._grad
        v = scipy.linalg.solve_triangular(
            self._L, self._sqrtW[:, None] * Ks.T, lower=True)
        f_var = np.maximum(self._kernel.diag(X) - (v ** 2).sum(axis=0), 0)
        # probit-style correction for the logistic link
        kappa = 1.0 / np.sqrt(1.0 + np.pi * f_var / 8)
        p = sigmoid(kappa * f_mean)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]


__all__ = ["Kernel", "ConstantKernel", "WhiteKernel", "RBF", "Matern",
           "RationalQuadratic", "ExpSineSquared", "DotProduct", "Sum",
           "Product", "ConstantTimes", "GaussianProcessRegressor",
           "GaussianProcessClassifier"]
