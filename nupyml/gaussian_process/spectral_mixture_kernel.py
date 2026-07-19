"""A kernel learned in the FREQUENCY domain (Wilson & Adams, 2013)."""
import numpy as np
from .kernels import Kernel, RBF


class SpectralMixtureKernel(Kernel):
    """A kernel learned in the FREQUENCY domain (Wilson & Adams, 2013).

    THE IDEA
    --------
    By Bochner's theorem, every stationary kernel is the Fourier transform of a
    spectral density. RBF corresponds to a single Gaussian centred at frequency 0
    -- so it can only model smooth, aperiodic structure. A spectral MIXTURE models
    the spectral density as a sum of ``Q`` Gaussians at learnable frequencies,
    giving the kernel::

        k(tau) = sum_q w_q * exp(-2 pi^2 tau^2 v_q) * cos(2 pi tau mu_q)

    The non-zero centre frequencies ``mu_q`` let it represent PERIODIC and
    quasi-periodic patterns and, crucially, EXTRAPOLATE them -- something RBF (which
    reverts to the mean away from data) cannot. The kernel is expressive enough to
    approximate any stationary kernel given enough components.
    """

    def __init__(self, weights=None, means=None, variances=None, n_components=3):
        self.n_components = n_components
        self.weights = weights
        self.means = means
        self.variances = variances

    def _params(self):
        q = self.n_components
        w = np.ones(q) / q if self.weights is None else np.asarray(self.weights)
        mu = (np.linspace(0.1, 1.0, q) if self.means is None
              else np.asarray(self.means))
        v = np.ones(q) * 0.5 if self.variances is None else np.asarray(self.variances)
        return w, mu, v

    def __call__(self, A, B):
        A = np.asarray(A, float); B = np.asarray(B, float)
        tau = A[:, None, :] - B[None, :, :]            # (n, m, d)
        w, mu, v = self._params()
        K = np.zeros((len(A), len(B)))
        for q in range(self.n_components):
            comp = np.exp(-2 * np.pi ** 2 * tau ** 2 * v[q]) * \
                np.cos(2 * np.pi * tau * mu[q])
            K += w[q] * comp.prod(axis=2)              # product over dimensions
        return K

    def diag(self, A):
        w, _, _ = self._params()
        return np.full(len(A), w.sum())

    @property
    def theta(self):
        w, mu, v = self._params()
        return np.log(np.concatenate([w, mu, v]) + 1e-8)

    @theta.setter
    def theta(self, t):
        vals = np.exp(np.asarray(t))
        q = self.n_components
        self.weights, self.means, self.variances = vals[:q], vals[q:2 * q], vals[2 * q:]

    def clone(self):
        return SpectralMixtureKernel(self.weights, self.means, self.variances,
                                     self.n_components)


__all__ = ["SpectralMixtureKernel"]
