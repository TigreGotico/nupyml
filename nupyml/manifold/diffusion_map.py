"""Embed with the eigenvectors of a random walk on the data (Coifman, 2006)."""
import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist, squareform
from ..base import BaseEstimator, TransformerMixin
from ..utils import check_array


def _diffusion_operator(X, epsilon=None, alpha=1.0, knn_bandwidth=None):
    """Build a Markov transition matrix P from an anisotropic Gaussian kernel."""
    D2 = cdist(X, X, "sqeuclidean")
    if knn_bandwidth is not None:                    # adaptive per-point bandwidth
        k = min(knn_bandwidth, X.shape[0] - 1)
        sigma = np.sort(np.sqrt(D2), axis=1)[:, k][:, None]
        sigma = np.maximum(sigma, 1e-12)
        W = np.exp(-D2 / (sigma * sigma.T))
    else:
        if epsilon is None:
            epsilon = np.median(D2[D2 > 0])
        W = np.exp(-D2 / epsilon)
    # alpha-normalisation removes the effect of sampling density (alpha=1: Laplace-
    # Beltrami on the underlying manifold, regardless of how it was sampled).
    q = W.sum(axis=1)
    K = W / (q[:, None] ** alpha * q[None, :] ** alpha)
    d = K.sum(axis=1)
    P = K / d[:, None]
    return P


class DiffusionMap(BaseEstimator, TransformerMixin):
    """Embed with the eigenvectors of a random walk on the data (Coifman, 2006).

    Run a random walk that only steps between nearby points. After ``t`` steps the
    probability of walking from one point to another is small unless a short PATH
    of near-neighbours connects them -- so the "diffusion distance" measures
    connectivity along the manifold, not the straight-line distance that fools
    Isomap across gaps. Those distances are captured exactly by the leading
    eigenvectors of the transition matrix, scaled by their eigenvalue raised to
    the diffusion time ``t``. Increasing ``t`` zooms out to coarser structure.
    """

    def __init__(self, n_components=2, t=1, epsilon=None, alpha=1.0):
        self.n_components = n_components
        self.t = t
        self.epsilon = epsilon
        self.alpha = alpha

    def fit_transform(self, X, y=None):
        X = check_array(X)
        P = _diffusion_operator(X, self.epsilon, self.alpha)
        # symmetric conjugate Ms = D^{1/2} P D^{-1/2} shares P's spectrum but is
        # symmetric, so eigh gives real, orthonormal eigenvectors cheaply.
        pi = P.sum(axis=0)
        s = np.sqrt(pi)
        Ms = (s[:, None] * P) / s[None, :]
        Ms = (Ms + Ms.T) / 2
        vals, vecs = scipy.linalg.eigh(Ms)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        psi = vecs / s[:, None]                       # back to the walk's eigenvectors
        # drop the trivial constant eigenvector (eigenvalue 1), scale by lambda^t
        emb = psi[:, 1:self.n_components + 1] * (vals[1:self.n_components + 1] ** self.t)
        self.embedding_ = emb
        self.eigenvalues_ = vals
        return emb

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self


__all__ = ["DiffusionMap"]
