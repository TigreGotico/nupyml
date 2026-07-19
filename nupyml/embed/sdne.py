"""SDNE: structural deep network embedding via a proximity-preserving autoencoder."""
import numpy as np

from ..base import BaseEstimator
from ..autograd import Tensor
from ..nn.module import Module, Parameter
from ..nn.layers import Linear
from ..utils import check_random_state


class SDNE(BaseEstimator):
    """Embed a graph with a DEEP autoencoder on its adjacency (Wang et al., 2016).

    Shallow graph embeddings (a single factorisation) cannot capture the highly
    non-linear structure of real networks. SDNE feeds each node's ADJACENCY ROW -- who
    it connects to -- through a deep autoencoder. Reconstructing that row forces the
    bottleneck to preserve SECOND-order proximity (nodes with similar neighbourhoods get
    similar codes), while an extra Laplacian penalty pulls DIRECTLY connected nodes'
    codes together for FIRST-order proximity. Reconstruction errors on present edges are
    up-weighted so the model does not just predict "no edge" everywhere. The bottleneck
    activations are the embeddings. ``dim`` is the embedding size; ``beta`` up-weights
    observed edges; ``alpha`` weights the first-order term.
    """

    def __init__(self, dim=16, hidden=64, epochs=200, lr=0.01, alpha=0.2, beta=5.0,
                 random_state=None):
        self.dim = dim
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state

    def fit(self, adjacency):
        A = np.asarray(adjacency, float)
        N = len(A)
        rng = check_random_state(self.random_state)

        class _AE(Module):
            def __init__(s):
                super().__init__()
                s.e1 = Linear(N, self.hidden, rng=rng)
                s.e2 = Linear(self.hidden, self.dim, rng=rng)
                s.d1 = Linear(self.dim, self.hidden, rng=rng)
                s.d2 = Linear(self.hidden, N, rng=rng)

            def encode(s, x):
                return s.e2(s.e1(x).relu())

            def forward(s, x):
                z = s.encode(x)
                return z, s.d2(s.d1(z).relu())

        ae = _AE()
        params = ae.parameters()
        X = Tensor(A)
        # up-weight reconstruction on observed edges (B); Laplacian for 1st-order term
        B = Tensor(1.0 + (self.beta - 1.0) * (A > 0))
        deg = np.diag(A.sum(axis=1))
        L = Tensor(deg - A)
        for _ in range(self.epochs):
            z, recon = ae(X)
            recon_err = (((recon - X) * B) ** 2).sum()      # 2nd-order proximity
            # 1st-order: trace(z^T L z) = sum over edges ||z_i - z_j||^2
            first = (z * (L @ z)).sum()
            loss = recon_err + self.alpha * first
            for p in params:
                p.zero_grad()
            loss.backward()
            for p in params:
                p.data -= self.lr * p.grad / N
        self.model_ = ae
        self.embedding_ = ae.encode(X).data
        return self

    def transform(self):
        return self.embedding_


__all__ = ["SDNE"]
