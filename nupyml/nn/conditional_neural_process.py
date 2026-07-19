"""Conditional Neural Process: amortised regression over a family of functions."""
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


class ConditionalNeuralProcess(Module):
    """Learn to regress a whole FAMILY of functions, with uncertainty
    (Garnelo et al., 2018).

    A Gaussian process regresses one function beautifully but costs ``O(n^3)`` and
    must be re-fit per dataset. A Conditional Neural Process AMORTISES that: an encoder
    maps each observed (x, y) CONTEXT point to a vector, averages them into a single
    representation of "which function is this", and a decoder predicts the mean AND
    variance at any target x from that representation. Trained across many functions,
    it produces GP-like predictions -- uncertainty that shrinks where context is dense
    -- in a single forward pass. ``forward`` takes context (x, y) and target x.
    """

    def __init__(self, x_dim=1, y_dim=1, hidden=32, r_dim=32, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.encoder = _mlp([x_dim + y_dim, hidden, r_dim], r)
        self.decoder = _mlp([r_dim + x_dim, hidden, 2 * y_dim], r)
        self.y_dim = y_dim

    def parameters(self):
        return [p for l in (self.encoder + self.decoder) for p in l.parameters()]

    def forward(self, cx, cy, tx):
        cx = Tensor._wrap(cx); cy = Tensor._wrap(cy); tx = Tensor._wrap(tx)
        r = _run(self.encoder, Tensor.concatenate([cx, cy], axis=1))
        r = r.mean(axis=0, keepdims=True)                 # aggregate the context
        r_rep = Tensor(np.tile(r.data, (tx.shape[0], 1)))
        out = _run(self.decoder, Tensor.concatenate([r_rep, tx], axis=1))
        mean = out[:, :self.y_dim]
        log_sigma = out[:, self.y_dim:]
        sigma = (0.1 + (log_sigma * 0.5).exp())           # positive std
        return mean, sigma

    def loss(self, cx, cy, tx, ty):
        mean, sigma = self.forward(cx, cy, tx)
        ty = Tensor._wrap(ty)
        return (0.5 * (((ty - mean) / sigma) ** 2) + sigma.log()).mean()


__all__ = ["ConditionalNeuralProcess"]
