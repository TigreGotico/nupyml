"""Diffusion-based manifold learning: diffusion maps and PHATE.

Both build a random walk on the data graph and let it run. Where Laplacian
eigenmaps (``SpectralEmbedding``) use the graph Laplacian directly, these use the
DIFFUSION operator -- the transition matrix of a random walk -- whose powers
reveal structure at successively coarser scales. Diffusion maps embed with its
eigenvectors; PHATE turns the diffused probabilities into a distance and lays
that out, which is what makes it preserve both local and global trajectory
structure so well on biological data.
"""
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


__all__ = ["DiffusionMap", "PHATE"]
