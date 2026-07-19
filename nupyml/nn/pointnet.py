"""PointNet: a permutation-invariant network over an unordered point set."""
import numpy as np

from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear


def _mlp(dims, rng):
    return [Linear(dims[i], dims[i + 1], rng=rng) for i in range(len(dims) - 1)]


def _run(layers, x, act=True):
    h = x
    for i, l in enumerate(layers):
        h = l(h)
        if act and i < len(layers) - 1:
            h = h.relu()
    return h


class PointNet(Module):
    """A network invariant to the ORDER of its inputs (Qi et al., 2017).

    A point cloud has no canonical ordering -- the same shape can be listed in any
    permutation -- so a network reading it must give the SAME answer regardless of
    order. PointNet does this with a shared per-point MLP followed by a SYMMETRIC
    aggregation (max-pooling over points): each point is embedded independently, then
    a permutation-invariant pool collapses them to one global descriptor that a
    classifier reads. Simple, and it was the breakthrough that let deep nets consume
    raw 3-D points. Input ``(batch, n_points, point_dim)``.
    """

    def __init__(self, point_dim=3, n_classes=2, hidden=64, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.point_mlp = _mlp([point_dim, hidden, hidden], r)
        self.head = _mlp([hidden, hidden, n_classes], r)

    def parameters(self):
        return [p for l in (self.point_mlp + self.head) for p in l.parameters()]

    def forward(self, clouds):
        x = Tensor._wrap(clouds)
        B, N, D = x.shape
        feats = _run(self.point_mlp, x.reshape(B * N, D)).reshape(B, N, -1)
        pooled = feats.max(axis=1)                         # symmetric pool -> invariant
        return _run(self.head, pooled)


__all__ = ["PointNet"]
