"""Permutation-INVARIANT network over a set (Zaheer et al., 2017)."""
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


class DeepSets(Module):
    """Permutation-INVARIANT network over a set (Zaheer et al., 2017).

    A function of a SET must not depend on the order of its elements. DeepSets
    guarantees that structurally: apply the same ``phi`` to every element, POOL
    the results with a symmetric operation (sum), then apply ``rho``::

        f({x_1..x_n}) = rho( sum_i phi(x_i) )

    Any permutation-invariant function has this form, so it is not just a heuristic
    but the general recipe. Input is ``(batch, set_size, features)``; output is
    ``(batch, out_dim)``.
    """

    def __init__(self, in_dim, hidden=64, out_dim=1, rng=None):
        super().__init__()
        self.phi = _Seq(_mlp([in_dim, hidden, hidden], rng))
        self.rho = _Seq(_mlp([hidden, hidden, out_dim], rng))

    def parameters(self):
        return self.phi.parameters() + self.rho.parameters()

    def forward(self, x):
        x = Tensor._wrap(x)
        h = self.phi(x)                               # (b, n, hidden)
        pooled = h.sum(axis=1)                         # symmetric pool -> (b, hidden)
        return self.rho(pooled)


__all__ = ["DeepSets"]
