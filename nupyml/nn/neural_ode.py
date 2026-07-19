"""Continuous-depth network: the hidden state evolves by an ODE (Chen, 2018)."""
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


class NeuralODE(Module):
    """Continuous-depth network: the hidden state evolves by an ODE (Chen, 2018).

    A residual block computes ``h + f(h)``; stack many and you are doing Euler
    steps of an ODE ``dh/dt = f(h)``. Neural ODEs take that limit literally -- the
    "depth" becomes continuous TIME, and the output is the ODE solution at ``t=1``,
    produced by a numerical solver (RK4 here). This gives constant memory
    (backprop through the solver) and adaptive depth. ``forward`` integrates the
    input state from 0 to 1.
    """

    def __init__(self, dim, hidden=32, n_steps=10, rng=None):
        super().__init__()
        self.func = _Seq(_mlp([dim, hidden, dim], rng))
        self.n_steps = n_steps

    def parameters(self):
        return self.func.parameters()

    def forward(self, x):
        h = Tensor._wrap(x)
        dt = 1.0 / self.n_steps
        for _ in range(self.n_steps):                  # fixed-step RK4 integration
            k1 = self.func(h)
            k2 = self.func(h + 0.5 * dt * k1)
            k3 = self.func(h + 0.5 * dt * k2)
            k4 = self.func(h + dt * k3)
            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h


__all__ = ["NeuralODE"]
