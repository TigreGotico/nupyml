"""Gaussian-process expansion: a frequency-domain kernel, a heavy-tailed process,
and a deep-kernel GP.

The base package has exact GPR/GPC, composable kernels, and (v1) sparse/multi-
output GPs. These add an expressive learned kernel, a robust process, and a GP
with a neural feature map.
"""
import numpy as np

from .kernels import Kernel, RBF
from ..base import BaseEstimator, RegressorMixin, check_is_fitted
from ..utils import check_X_y, check_array


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


class DeepKernelGP(BaseEstimator, RegressorMixin):
    """A GP on features learned by a NEURAL NET (deep kernel learning, Wilson 2016).

    A GP's power is limited by its kernel: RBF measures similarity in the RAW input
    space, so it struggles when "similar for the task" is a nonlinear function of
    the inputs. Deep kernel learning puts a neural feature extractor BEFORE the
    kernel -- ``k(x, x') = k_RBF(g(x), g(x'))`` -- so the GP measures similarity in a
    learned, task-relevant space while keeping the GP's calibrated uncertainty. Here
    the feature net is trained on the regression target, then an exact GP is fit on
    its features (a practical, non-end-to-end deep kernel).
    """

    def __init__(self, hidden=(32, 16), feature_dim=8, alpha=1e-4, epochs=200,
                 lr=0.01, random_state=None):
        self.hidden = hidden
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state

    def fit(self, X, y):
        from .. import nn
        from ..autograd import Tensor
        from . import GaussianProcessRegressor
        X, y = check_X_y(X, y, y_numeric=True)
        dims = [X.shape[1], *self.hidden, self.feature_dim]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.net_ = nn.Sequential(*layers[:-1])       # feature extractor
        head = nn.Linear(self.feature_dim, 1)
        opt = nn.Adam(list(self.net_.parameters()) + list(head.parameters()),
                      lr=self.lr)
        yt = Tensor(y.reshape(-1, 1))
        for _ in range(self.epochs):                  # train the features on y
            opt.zero_grad()
            pred = head(self.net_(Tensor(X)))
            ((pred - yt) ** 2).mean().backward()
            opt.step()
        feats = self.net_(Tensor(X)).data
        self.gp_ = GaussianProcessRegressor(kernel=RBF(), alpha=self.alpha,
                                            optimize=False).fit(feats, y)
        return self

    def _features(self, X):
        from ..autograd import Tensor
        return self.net_(Tensor(check_array(X))).data

    def predict(self, X, return_std=False):
        check_is_fitted(self, "gp_")
        return self.gp_.predict(self._features(X), return_std=return_std)


__all__ = ["SpectralMixtureKernel", "StudentTProcess", "DeepKernelGP"]
