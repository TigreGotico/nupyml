"""SimSiam: non-contrastive self-supervised learning with a stop-gradient."""
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


class SimSiam(Module):
    """Self-supervised learning with NO negatives, no collapse (Chen & He, 2021).

    Contrastive methods need many negative pairs to stop the network mapping
    everything to one point. SimSiam shows you can drop the negatives entirely: it
    feeds two AUGMENTED views through the same encoder and a projector, then a small
    PREDICTOR on one side must match the other side's (STOP-GRADIENT) projection.
    The stop-gradient is the whole trick -- it stops the trivial collapse the missing
    negatives would otherwise allow -- and the network learns useful invariant
    representations. Encoder + projector + predictor MLPs here.
    """

    def __init__(self, in_dim, hidden=64, proj_dim=32, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.encoder = _mlp([in_dim, hidden, hidden], r)
        self.projector = _mlp([hidden, proj_dim], r)
        self.predictor = _mlp([proj_dim, proj_dim // 2, proj_dim], r)

    def parameters(self):
        return [p for l in (self.encoder + self.projector + self.predictor)
                for p in l.parameters()]

    def encode(self, x):
        return _run(self.encoder, Tensor._wrap(x))

    def _proj(self, x):
        return _run(self.projector, self.encode(x))

    def _neg_cos(self, p, z):
        p = p / ((p * p).sum(axis=1, keepdims=True) ** 0.5 + 1e-8)
        z = z / ((z * z).sum(axis=1, keepdims=True) ** 0.5 + 1e-8)
        return -(p * z).sum(axis=1).mean()

    def loss(self, view1, view2):
        z1, z2 = self._proj(view1), self._proj(view2)
        p1 = _run(self.predictor, z1); p2 = _run(self.predictor, z2)
        # symmetric loss with stop-gradient on the target branch
        return 0.5 * (self._neg_cos(p1, z2.detach()) + self._neg_cos(p2, z1.detach()))


__all__ = ["SimSiam"]
