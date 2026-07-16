"""Composable GP kernels with log-space hyperparameters.

Every kernel exposes ``theta`` (a log-space vector of its hyperparameters) so
the marginal-likelihood optimizer can treat any composition uniformly.
"""
import numpy as np
from scipy.spatial.distance import cdist


class Kernel:
    """Base class: kernels compose with + and *, and scale with a float."""

    def __call__(self, A, B):
        raise NotImplementedError

    def diag(self, A):
        return np.diag(self(A, A)).copy()

    @property
    def theta(self):
        raise NotImplementedError

    @theta.setter
    def theta(self, value):
        raise NotImplementedError

    @property
    def n_dims(self):
        return len(self.theta)

    def clone(self):
        raise NotImplementedError

    def __add__(self, other):
        return Sum(self, other if isinstance(other, Kernel) else ConstantKernel(other))

    __radd__ = __add__

    def __mul__(self, other):
        return Product(self, other if isinstance(other, Kernel)
                       else ConstantKernel(other))

    __rmul__ = __mul__

    def __repr__(self):
        return f"{type(self).__name__}({np.round(np.exp(self.theta), 3).tolist()})"


class ConstantKernel(Kernel):
    def __init__(self, constant_value=1.0):
        self.constant_value = constant_value

    def __call__(self, A, B):
        return np.full((len(A), len(B)), self.constant_value)

    @property
    def theta(self):
        return np.array([np.log(self.constant_value)])

    @theta.setter
    def theta(self, t):
        self.constant_value = float(np.exp(t[0]))

    def clone(self):
        return ConstantKernel(self.constant_value)


class WhiteKernel(Kernel):
    """Independent noise: only contributes on the diagonal of K(X, X)."""

    def __init__(self, noise_level=1.0):
        self.noise_level = noise_level

    def __call__(self, A, B):
        if A is B or (A.shape == B.shape and np.array_equal(A, B)):
            return self.noise_level * np.eye(len(A))
        return np.zeros((len(A), len(B)))

    def diag(self, A):
        return np.full(len(A), self.noise_level)

    @property
    def theta(self):
        return np.array([np.log(self.noise_level)])

    @theta.setter
    def theta(self, t):
        self.noise_level = float(np.exp(t[0]))

    def clone(self):
        return WhiteKernel(self.noise_level)


class RBF(Kernel):
    """Squared exponential. A vector length_scale gives per-dimension ARD."""

    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    @property
    def _anisotropic(self):
        return np.ndim(self.length_scale) > 0

    def _scaled(self, A, B):
        ls = np.asarray(self.length_scale, dtype=np.float64)
        return A / ls, B / ls

    def __call__(self, A, B):
        As, Bs = self._scaled(A, B)
        return np.exp(-0.5 * cdist(As, Bs, "sqeuclidean"))

    def diag(self, A):
        return np.ones(len(A))

    @property
    def theta(self):
        return np.log(np.atleast_1d(np.asarray(self.length_scale,
                                               dtype=np.float64)))

    @theta.setter
    def theta(self, t):
        vals = np.exp(np.asarray(t, dtype=np.float64))
        self.length_scale = vals if self._anisotropic else float(vals[0])

    def clone(self):
        ls = self.length_scale
        return RBF(np.array(ls, copy=True) if np.ndim(ls) else ls)


class Matern(Kernel):
    def __init__(self, length_scale=1.0, nu=1.5):
        self.length_scale = length_scale
        self.nu = nu

    @property
    def _anisotropic(self):
        return np.ndim(self.length_scale) > 0

    def __call__(self, A, B):
        ls = np.asarray(self.length_scale, dtype=np.float64)
        d = cdist(A / ls, B / ls)
        if self.nu == 0.5:
            return np.exp(-d)
        if self.nu == 1.5:
            s = np.sqrt(3) * d
            return (1 + s) * np.exp(-s)
        if self.nu == 2.5:
            s = np.sqrt(5) * d
            return (1 + s + s ** 2 / 3) * np.exp(-s)
        if self.nu == np.inf:
            return np.exp(-0.5 * d ** 2)
        # general nu through the modified Bessel function
        from scipy.special import kv, gamma
        d = np.maximum(d, 1e-12)
        tmp = np.sqrt(2 * self.nu) * d
        return (2 ** (1 - self.nu) / gamma(self.nu)) * tmp ** self.nu * kv(self.nu, tmp)

    def diag(self, A):
        return np.ones(len(A))

    @property
    def theta(self):
        return np.log(np.atleast_1d(np.asarray(self.length_scale,
                                               dtype=np.float64)))

    @theta.setter
    def theta(self, t):
        vals = np.exp(np.asarray(t, dtype=np.float64))
        self.length_scale = vals if self._anisotropic else float(vals[0])

    def clone(self):
        ls = self.length_scale
        return Matern(np.array(ls, copy=True) if np.ndim(ls) else ls, self.nu)


class RationalQuadratic(Kernel):
    """Scale mixture of RBFs; alpha -> inf recovers the RBF."""

    def __init__(self, length_scale=1.0, alpha=1.0):
        self.length_scale = length_scale
        self.alpha = alpha

    def __call__(self, A, B):
        d2 = cdist(A, B, "sqeuclidean")
        return (1 + d2 / (2 * self.alpha * self.length_scale ** 2)) ** (-self.alpha)

    def diag(self, A):
        return np.ones(len(A))

    @property
    def theta(self):
        return np.log([self.length_scale, self.alpha])

    @theta.setter
    def theta(self, t):
        self.length_scale, self.alpha = np.exp(t[0]), np.exp(t[1])

    def clone(self):
        return RationalQuadratic(self.length_scale, self.alpha)


class ExpSineSquared(Kernel):
    """Periodic kernel."""

    def __init__(self, length_scale=1.0, periodicity=1.0):
        self.length_scale = length_scale
        self.periodicity = periodicity

    def __call__(self, A, B):
        d = cdist(A, B, "euclidean")
        arg = np.pi * d / self.periodicity
        return np.exp(-2 * (np.sin(arg) / self.length_scale) ** 2)

    def diag(self, A):
        return np.ones(len(A))

    @property
    def theta(self):
        return np.log([self.length_scale, self.periodicity])

    @theta.setter
    def theta(self, t):
        self.length_scale, self.periodicity = np.exp(t[0]), np.exp(t[1])

    def clone(self):
        return ExpSineSquared(self.length_scale, self.periodicity)


class DotProduct(Kernel):
    """Linear (non-stationary) kernel."""

    def __init__(self, sigma_0=1.0):
        self.sigma_0 = sigma_0

    def __call__(self, A, B):
        return self.sigma_0 ** 2 + A @ B.T

    def diag(self, A):
        return self.sigma_0 ** 2 + (A ** 2).sum(axis=1)

    @property
    def theta(self):
        return np.array([np.log(self.sigma_0)])

    @theta.setter
    def theta(self, t):
        self.sigma_0 = float(np.exp(t[0]))

    def clone(self):
        return DotProduct(self.sigma_0)


class _Operator(Kernel):
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    @property
    def theta(self):
        return np.concatenate([self.k1.theta, self.k2.theta])

    @theta.setter
    def theta(self, t):
        n1 = self.k1.n_dims
        self.k1.theta = t[:n1]
        self.k2.theta = t[n1:]

    def clone(self):
        return type(self)(self.k1.clone(), self.k2.clone())

    def __repr__(self):
        op = "+" if isinstance(self, Sum) else "*"
        return f"({self.k1!r} {op} {self.k2!r})"


class Sum(_Operator):
    def __call__(self, A, B):
        return self.k1(A, B) + self.k2(A, B)

    def diag(self, A):
        return self.k1.diag(A) + self.k2.diag(A)


class Product(_Operator):
    def __call__(self, A, B):
        return self.k1(A, B) * self.k2(A, B)

    def diag(self, A):
        return self.k1.diag(A) * self.k2.diag(A)


__all__ = ["Kernel", "ConstantKernel", "WhiteKernel", "RBF", "Matern",
           "RationalQuadratic", "ExpSineSquared", "DotProduct", "Sum", "Product"]
