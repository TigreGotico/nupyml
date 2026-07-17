"""Scalable and multi-output Gaussian processes.

Exact GP regression costs O(n^3) (it inverts an n x n kernel matrix), which caps
it at a few thousand points and one output. These two lift those limits: inducing
points for scale, and per-output models for vector targets.
"""
import numpy as np

from ..base import BaseEstimator, RegressorMixin, clone, check_is_fitted
from ..utils import check_array, check_X_y, check_random_state


class SparseGaussianProcessRegressor(BaseEstimator, RegressorMixin):
    """Sparse GP via INDUCING POINTS (Subset-of-Regressors approximation).

    THE SCALING PROBLEM
    -------------------
    Exact GP regression inverts the ``n x n`` kernel matrix -- ``O(n^3)`` time,
    ``O(n^2)`` memory -- so it dies past a few thousand points. The fix: summarise
    the whole training set through ``m << n`` INDUCING points and let all
    predictions flow through them. The cost drops to ``O(n m^2)``.

    THE APPROXIMATION
    -----------------
    Subset of Regressors approximates every kernel entry by routing it through the
    inducing set, ``K(a,b) ≈ K(a,U) K(U,U)^{-1} K(U,b)``. The posterior mean at a
    test point becomes::

        mean(x*) = K(x*,U) Sigma K(U,X) y / sigma^2,
        Sigma = (K(U,U) + K(U,X) K(X,U) / sigma^2)^{-1}

    -- everything is ``m x m`` or ``m x n``, never ``n x n``. The inducing points
    are placed by k-means on the inputs (a cheap, effective heuristic; the full
    method would optimise them). The trade is a low-rank approximation to the true
    covariance, which can UNDER-estimate predictive variance far from the inducing
    points -- the known caveat of SoR.
    """

    def __init__(self, kernel=None, n_inducing=50, alpha=1e-6, random_state=None):
        self.kernel = kernel
        self.n_inducing = n_inducing
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        from . import RBF
        from ..cluster import KMeans
        X, y = check_X_y(X, y)
        self.kernel_ = self.kernel if self.kernel is not None else RBF()
        m = min(self.n_inducing, len(X))
        self.U_ = KMeans(n_clusters=m, random_state=self.random_state).fit(X).cluster_centers_
        Kuu = self.kernel_(self.U_, self.U_) + 1e-8 * np.eye(m)
        Kux = self.kernel_(self.U_, X)
        s2 = self.alpha
        Sigma = np.linalg.inv(Kuu + Kux @ Kux.T / s2)
        self.Sigma_ = Sigma
        self.w_ = Sigma @ Kux @ y / s2               # cached predictor weights
        self._Kuu = Kuu
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self, "w_")
        X = check_array(X)
        Ksu = self.kernel_(X, self.U_)
        mean = Ksu @ self.w_
        if not return_std:
            return mean
        # SoR predictive variance uses the inducing-point projection
        var = np.einsum("ij,jk,ik->i", Ksu, self.Sigma_, Ksu)
        return mean, np.sqrt(np.maximum(var, 0))


class MultiOutputGaussianProcessRegressor(BaseEstimator, RegressorMixin):
    """A GP for VECTOR targets: one GP per output, sharing the kernel.

    Exact GP regression predicts a scalar. The simplest honest multi-output GP
    fits an independent GP to each target dimension with a shared kernel form --
    valid when the outputs are conditionally independent given the inputs (a
    diagonal coregionalization). Predictions and per-output uncertainties come
    back stacked. When outputs are CORRELATED, the intrinsic-coregionalization
    model (a shared latent GP mixed across outputs) is the extension; this is its
    independent-output special case, which is often enough and always a fair
    baseline.
    """

    def __init__(self, kernel=None, alpha=1e-6, random_state=None):
        self.kernel = kernel
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, Y):
        from . import GaussianProcessRegressor
        X = check_array(X)
        Y = check_array(Y)
        self.models_ = []
        for k in range(Y.shape[1]):
            gp = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha,
                                          optimize=False,
                                          random_state=self.random_state)
            gp.fit(X, Y[:, k])
            self.models_.append(gp)
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self, "models_")
        X = check_array(X)
        if not return_std:
            return np.column_stack([m.predict(X) for m in self.models_])
        means, stds = [], []
        for m in self.models_:
            mu, sd = m.predict(X, return_std=True)
            means.append(mu); stds.append(sd)
        return np.column_stack(means), np.column_stack(stds)


__all__ = ["SparseGaussianProcessRegressor",
           "MultiOutputGaussianProcessRegressor"]
