"""Diffuse, take the potential, then lay it out (Moon et al., 2019)."""
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


class PHATE(BaseEstimator, TransformerMixin):
    """Diffuse, take the potential, then lay it out (Moon et al., 2019).

    Diffusion distances collapse to noise at long times because the walk forgets
    where it started. PHATE fixes this by taking the ``-log`` of the diffused
    probabilities first -- the "potential" -- which keeps far-apart points
    informatively separated, then embeds those potential distances with MDS. The
    payoff is an embedding that preserves BOTH local neighbourhoods and global
    branching/trajectory structure in one picture, which is why it became a
    standard for single-cell data. ``knn`` sets the adaptive bandwidth.
    """

    def __init__(self, n_components=2, t=5, knn=5):
        self.n_components = n_components
        self.t = t
        self.knn = knn

    def fit_transform(self, X, y=None):
        X = check_array(X)
        P = _diffusion_operator(X, knn_bandwidth=self.knn)
        Pt = np.linalg.matrix_power(P, self.t)        # diffuse t steps
        potential = -np.log(Pt + 1e-12)               # informative-distance transform
        # potential distance = euclidean distance between rows of the potential
        Dpot = cdist(potential, potential)
        self.embedding_ = self._classical_mds(Dpot)
        return self.embedding_

    def _classical_mds(self, D):
        n = D.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J                   # double-centre
        vals, vecs = scipy.linalg.eigh(B)
        order = np.argsort(vals)[::-1][:self.n_components]
        L = np.sqrt(np.clip(vals[order], 0, None))
        return vecs[:, order] * L

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self


__all__ = ["PHATE"]
