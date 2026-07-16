"""Gaussian process regression and classification."""
import numpy as np
import scipy.linalg
import scipy.optimize
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin, check_is_fitted
from ..preprocessing import LabelEncoder
from ..utils import check_X_y, check_array, sigmoid


class RBF:
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def __call__(self, A, B):
        return np.exp(-0.5 * cdist(A, B, "sqeuclidean") / self.length_scale ** 2)

    @property
    def theta(self):
        return np.array([np.log(self.length_scale)])

    @theta.setter
    def theta(self, t):
        self.length_scale = float(np.exp(t[0]))

    def clone(self):
        return RBF(self.length_scale)


class Matern:
    """Matern kernel with nu in {0.5, 1.5, 2.5}."""

    def __init__(self, length_scale=1.0, nu=1.5):
        self.length_scale = length_scale
        self.nu = nu

    def __call__(self, A, B):
        d = cdist(A, B) / self.length_scale
        if self.nu == 0.5:
            return np.exp(-d)
        if self.nu == 1.5:
            s = np.sqrt(3) * d
            return (1 + s) * np.exp(-s)
        if self.nu == 2.5:
            s = np.sqrt(5) * d
            return (1 + s + s ** 2 / 3) * np.exp(-s)
        raise ValueError("nu must be 0.5, 1.5, or 2.5")

    @property
    def theta(self):
        return np.array([np.log(self.length_scale)])

    @theta.setter
    def theta(self, t):
        self.length_scale = float(np.exp(t[0]))

    def clone(self):
        return Matern(self.length_scale, self.nu)


class ConstantTimes:
    """scale^2 * kernel wrapper."""

    def __init__(self, kernel, scale=1.0):
        self.kernel = kernel
        self.scale = scale

    def __call__(self, A, B):
        return self.scale ** 2 * self.kernel(A, B)

    @property
    def theta(self):
        return np.r_[np.log(self.scale), self.kernel.theta]

    @theta.setter
    def theta(self, t):
        self.scale = float(np.exp(t[0]))
        self.kernel.theta = t[1:]

    def clone(self):
        return ConstantTimes(self.kernel.clone(), self.scale)


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
        kernel = (self.kernel or ConstantTimes(RBF(1.0), 1.0)).clone()
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
        var = np.maximum(np.diag(self.kernel_(X, X)) - (v ** 2).sum(axis=0), 0)
        return mean, np.sqrt(var)


class GaussianProcessClassifier(BaseEstimator, ClassifierMixin):
    """Binary GP classification via Laplace approximation (RBF kernel)."""

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
        kernel = self.kernel or ConstantTimes(RBF(1.0), 1.0)
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
        f_var = np.maximum(np.diag(self._kernel(X, X)) - (v ** 2).sum(axis=0), 0)
        # probit-style correction for the logistic link
        kappa = 1.0 / np.sqrt(1.0 + np.pi * f_var / 8)
        p = sigmoid(kappa * f_mean)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]


__all__ = ["RBF", "Matern", "ConstantTimes", "GaussianProcessRegressor",
           "GaussianProcessClassifier"]
