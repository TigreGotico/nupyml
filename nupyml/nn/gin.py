"""Graph Isomorphism Network -- the maximally-expressive message-passing GNN"""
import numpy as np
from ..autograd import Tensor
from .module import Module
from .layers import Linear, LayerNorm, ReLU, GELU


def _mlp(dims, rng):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [Linear(a, b, rng=rng), ReLU()]
    return layers[:-1]


class _Seq(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class GIN(Module):
    """Graph Isomorphism Network -- the maximally-expressive message-passing GNN
    (Xu et al., 2019).

    A GNN layer updates each node from its neighbours. GIN uses the aggregation
    that makes it as discriminative as the Weisfeiler-Lehman graph-isomorphism
    test (the theoretical ceiling for message passing): SUM the neighbours (sum
    injectively distinguishes multisets, where mean/max lose count information)
    and pass through an MLP::

        h_v <- MLP( (1 + eps) * h_v + sum_{u in N(v)} h_u )

    ``forward(X, A)`` takes node features ``(n, d)`` and an adjacency ``(n, n)``;
    ``graph_embedding`` sum-pools the final node features for graph-level tasks.
    """

    def __init__(self, in_dim, hidden=32, n_layers=2, eps=0.0, rng=None):
        super().__init__()
        self.eps = eps
        self.mlps = []
        d = in_dim
        for _ in range(n_layers):
            self.mlps.append(_Seq(_mlp([d, hidden, hidden], rng)))
            d = hidden

    def parameters(self):
        return [p for m in self.mlps for p in m.parameters()]

    def forward(self, X, A):
        h = Tensor._wrap(X)
        A = Tensor(np.asarray(A, float))
        for mlp in self.mlps:
            h = mlp((1 + self.eps) * h + A @ h)       # sum-aggregate + transform
        return h

    def graph_embedding(self, X, A):
        return self.forward(X, A).sum(axis=0)         # readout for the whole graph


__all__ = ["GIN"]
