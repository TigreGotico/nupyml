"""Energy-based model: learn an unnormalised density, sample it with Langevin dynamics."""
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


class EnergyBasedModel(Module):
    """Learn an unnormalised density and SAMPLE it with Langevin (LeCun; Du & Mordatch).

    A normalising flow must stay invertible; an autoregressive model must factorise.
    An energy-based model is free of both: it learns a scalar ENERGY ``E(x)`` that is
    LOW on data and high elsewhere -- the density is ``exp(-E)/Z`` with the intractable
    ``Z`` never computed. Training uses contrastive divergence: push DOWN the energy of
    real data and UP the energy of "negative" samples drawn by LANGEVIN dynamics
    (gradient descent on energy plus noise). Sampling is the same Langevin walk. The
    most flexible generative model, at the cost of MCMC in the loop.
    """

    def __init__(self, dim, hidden=64, langevin_steps=20, langevin_lr=0.1, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.net = _mlp([dim, hidden, hidden, 1], r)
        self.dim = dim
        self.langevin_steps = langevin_steps
        self.langevin_lr = langevin_lr

    def parameters(self):
        return [p for l in self.net for p in l.parameters()]

    def energy(self, x):
        return _run(self.net, Tensor._wrap(x)).reshape(-1)

    def sample(self, n, rng=None, init=None):
        rng = check_random_state(rng)
        x = rng.randn(n, self.dim) if init is None else np.array(init, float)
        for _ in range(self.langevin_steps):
            xt = Tensor(x, requires_grad=True)
            e = self.energy(xt).sum()
            e.backward()
            x = x - 0.5 * self.langevin_lr * xt.grad \
                + np.sqrt(self.langevin_lr) * rng.randn(*x.shape)   # Langevin step
            x = np.clip(x, -5, 5)
        return x

    def loss(self, x_data, reg=1.0, rng=None):
        x_data = Tensor._wrap(x_data)
        rng = check_random_state(rng)
        # contrastive divergence: start the Langevin chain AT the data (+ noise) --
        # far more stable than sampling from random noise
        init = x_data.data + 0.1 * rng.randn(*x_data.data.shape)
        neg = self.sample(len(x_data.data), rng=rng, init=init)
        e_pos = self.energy(x_data)
        e_neg = self.energy(neg)
        return (e_pos.mean() - e_neg.mean()
                + reg * ((e_pos ** 2).mean() + (e_neg ** 2).mean()))


__all__ = ["EnergyBasedModel"]
