"""Sparse variational Gaussian process: scalable GP regression via inducing points."""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class SparseVariationalGP(BaseEstimator):
    """Gaussian-process regression that SCALES via inducing points (Titsias, 2009).

    Exact GP regression costs ``O(n^3)`` -- hopeless past a few thousand points. A
    sparse GP summarises the data with ``m << n`` INDUCING points and does inference
    through them, dropping the cost to ``O(n m^2)``. The variational formulation
    chooses the inducing outputs to best approximate the true posterior (minimising a
    KL bound), so it keeps GP-quality predictive means AND uncertainty at a fraction of
    the cost. Inducing points placed by k-means here; RBF kernel. Predicts mean and
    standard deviation.
    """

    def __init__(self, n_inducing=10, length_scale=1.0, signal_var=1.0,
                 noise=0.1, random_state=None):
        self.n_inducing = n_inducing
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise = noise
        self.random_state = random_state

    def _kernel(self, A, B):
        from scipy.spatial.distance import cdist
        return self.signal_var * np.exp(
            -0.5 * cdist(A, B, "sqeuclidean") / self.length_scale ** 2)

    def fit(self, X, y):
        from ..cluster import KMeans
        X = check_array(X); y = np.asarray(y, float).ravel()
        rng = check_random_state(self.random_state)
        m = min(self.n_inducing, len(X))
        self.Z_ = KMeans(n_clusters=m, random_state=rng).fit(X).cluster_centers_
        Kmm = self._kernel(self.Z_, self.Z_) + 1e-6 * np.eye(m)
        Knm = self._kernel(X, self.Z_)
        Kmm_inv = np.linalg.inv(Kmm)
        # Titsias predictive: Sigma = (Kmm + Kmn Knm / noise^2)^{-1}
        sig = self.noise ** 2
        A = Kmm + Knm.T @ Knm / sig
        A_inv = np.linalg.inv(A + 1e-6 * np.eye(m))
        self.mu_ = A_inv @ Knm.T @ y / sig               # inducing-point mean
        self.Kmm_inv_ = Kmm_inv
        self.A_inv_ = A_inv
        return self

    def predict(self, X, return_std=False):
        X = check_array(X)
        Ksm = self._kernel(X, self.Z_)
        mean = Ksm @ self.mu_
        if not return_std:
            return mean
        Kss = np.diag(self._kernel(X, X))
        var = Kss - np.einsum("ij,jk,ik->i", Ksm, self.Kmm_inv_, Ksm) \
            + np.einsum("ij,jk,ik->i", Ksm, self.A_inv_, Ksm) + self.noise ** 2
        return mean, np.sqrt(np.maximum(var, 1e-9))


__all__ = ["NestedSampling", "DeterminantalPointProcess", "BayesianQuadrature",
           "SparseVariationalGP"]


__all__ = ["SparseVariationalGP"]
