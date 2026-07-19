"""Attention over a GRAPH, with edge features (Dwivedi & Bresson, 2020)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear


class GraphTransformer(Module):
    """Attention over a GRAPH, with edge features (Dwivedi & Bresson, 2020).

    A GNN aggregates from fixed neighbours with fixed weights; a transformer attends
    globally but ignores structure. The graph transformer merges them: each node
    attends to its NEIGHBOURS (masked by the adjacency), the attention score is
    modulated by the EDGE feature between them, and the usual residual + feed-forward
    follow. So it learns which neighbours matter, conditioned on how they are
    connected -- the backbone of modern molecular and graph-property models.
    ``forward(X, A, E)`` takes node features, adjacency, and an edge-feature matrix.
    """

    def __init__(self, dim, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.q = Linear(dim, dim, rng=r)
        self.k = Linear(dim, dim, rng=r)
        self.v = Linear(dim, dim, rng=r)
        self.edge = Linear(1, dim, rng=r)
        self.ff = Linear(dim, dim, rng=r)
        self.dim = dim

    def parameters(self):
        return (list(self.q.parameters()) + list(self.k.parameters())
                + list(self.v.parameters()) + list(self.edge.parameters())
                + list(self.ff.parameters()))

    def forward(self, X, A, E=None):
        X = Tensor._wrap(X)
        n = X.shape[0]
        A = np.asarray(A, float)
        Q, K, V = self.q(X), self.k(X), self.v(X)
        scores = (Q @ K.transpose()) * (1.0 / np.sqrt(self.dim))   # (n, n)
        if E is not None:                                # modulate by edge scalar
            scores = scores + Tensor(np.asarray(E, float))
        mask = np.where(A > 0, 0.0, -1e9)                # attend only to neighbours
        attn = F.softmax(scores + Tensor(mask), axis=1)
        out = attn @ V
        out = X + out                                    # residual
        return out + self.ff(out).relu()


__all__ = ["GraphTransformer"]
