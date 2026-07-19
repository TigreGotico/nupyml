"""Graph Attention Network v2 -- learn how much each neighbour matters"""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module, Parameter
from .layers import Linear, ReLU


class GATv2(Module):
    """Graph Attention Network v2 -- learn how much each neighbour matters
    (Brody et al., 2022).

    A GCN averages neighbours with fixed weights; GAT learns ATTENTION weights so a
    node listens more to relevant neighbours. GATv2 fixes the original GAT's flaw
    that its attention was "static" (the ranking of neighbours could not depend on
    the query node) by applying the nonlinearity BEFORE the attention vector, giving
    truly DYNAMIC attention. ``forward(X, A)`` takes node features ``(n, d)`` and an
    adjacency ``(n, n)`` and returns attention-aggregated features.
    """

    def __init__(self, in_dim, out_dim, rng=None):
        super().__init__()
        from ..utils import check_random_state
        r = check_random_state(rng)
        self.W = Linear(in_dim, out_dim, rng=r)
        self.a = Parameter(r.randn(2 * out_dim) * 0.1)   # attention vector

    def parameters(self):
        return list(self.W.parameters()) + [self.a]

    def forward(self, X, A):
        h = self.W(Tensor._wrap(X))                      # (n, out)
        n = h.shape[0]
        A = np.asarray(A)
        hd = h.data
        # GATv2 score: a . LeakyReLU(W h_i || W h_j)  -- nonlinearity before 'a'
        out_rows = []
        for i in range(n):
            nbrs = np.where(A[i] > 0)[0]
            if len(nbrs) == 0:
                out_rows.append(h[i:i + 1])
                continue
            scores = []
            for j in nbrs:
                cat = Tensor.concatenate([h[i:i + 1], h[j:j + 1]], axis=1)
                lr = cat.relu() + 0.01 * (cat - cat.relu())   # leaky relu
                scores.append((lr @ self.a.reshape(-1, 1)))
            s = Tensor.concatenate(scores, axis=0).reshape(1, -1)
            alpha = F.softmax(s, axis=1)                  # attention over neighbours
            agg = alpha @ h[nbrs]
            out_rows.append(agg)
        return Tensor.concatenate(out_rows, axis=0)


__all__ = ["GATv2"]
