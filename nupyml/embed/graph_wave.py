"""Embed a node's structural ROLE from heat diffusion (Donnat et al., 2018)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class GraphWave(BaseEstimator):
    """Embed a node's structural ROLE from heat diffusion (Donnat et al., 2018).

    Two nodes play the same structural role (both hubs, both bridge points) even if
    they are far apart -- proximity embeddings miss this. GraphWave treats each node
    as a heat source and lets heat DIFFUSE through the graph (a spectral graph
    wavelet); the resulting pattern of coefficients around a node is a fingerprint of
    its local structure. Summarising that pattern's distribution via its empirical
    CHARACTERISTIC FUNCTION (sampled at a few points) gives an embedding where
    structurally-equivalent nodes coincide, with no random walks or labels. ``scales``
    are the diffusion times.
    """

    def __init__(self, scales=(1.0,), n_samples=10, sample_max=2.0):
        self.scales = list(scales)
        self.n_samples = n_samples
        self.sample_max = sample_max

    def fit(self, adjacency):
        A = check_array(adjacency)
        n = len(A)
        deg = A.sum(axis=1)
        L = np.diag(deg) - A                             # unnormalised Laplacian
        vals, vecs = np.linalg.eigh(L)
        ts = np.linspace(0, self.sample_max, self.n_samples)
        embeddings = []
        for s in self.scales:
            # spectral graph wavelet operator: U exp(-s Lambda) U^T
            Psi = (vecs * np.exp(-s * vals)) @ vecs.T
            emb = np.zeros((n, 2 * self.n_samples))
            for node in range(n):
                coeffs = Psi[:, node]                    # wavelet centred at node
                # empirical characteristic function phi(t) = mean exp(i t coeff)
                for k, t in enumerate(ts):
                    emb[node, 2 * k] = np.mean(np.cos(t * coeffs))
                    emb[node, 2 * k + 1] = np.mean(np.sin(t * coeffs))
            embeddings.append(emb)
        self.embedding_ = np.hstack(embeddings)
        return self

    def get_vector(self, node):
        return self.embedding_[node]


__all__ = ["GraphWave"]
