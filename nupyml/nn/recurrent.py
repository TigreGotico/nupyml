"""Recurrent layers: vanilla RNN, GRU, LSTM."""
import numpy as np

from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, Parameter
from . import init


class _RecurrentBase(Module):
    """Processes (N, T, D) input, returns (all hidden states, final state)."""

    def __init__(self, input_size, hidden_size, n_gates, rng=None):
        super().__init__()
        rng = check_random_state(rng)
        g = n_gates
        self.hidden_size = hidden_size
        self.W_ih = Parameter(init.xavier_uniform((input_size, g * hidden_size), rng))
        self.W_hh = Parameter(init.xavier_uniform((hidden_size, g * hidden_size), rng))
        self.b = Parameter(np.zeros(g * hidden_size))

    def _init_state(self, n):
        return Tensor(np.zeros((n, self.hidden_size)))


class RNN(_RecurrentBase):
    def __init__(self, input_size, hidden_size, rng=None):
        super().__init__(input_size, hidden_size, n_gates=1, rng=rng)

    def forward(self, x, h0=None):
        n, T, _ = x.shape
        h = h0 if h0 is not None else self._init_state(n)
        outs = []
        for t in range(T):
            xt = x[:, t, :]
            h = (xt @ self.W_ih + h @ self.W_hh + self.b).tanh()
            outs.append(h)
        return Tensor.stack(outs, axis=1), h


class GRU(_RecurrentBase):
    def __init__(self, input_size, hidden_size, rng=None):
        super().__init__(input_size, hidden_size, n_gates=3, rng=rng)

    def forward(self, x, h0=None):
        n, T, _ = x.shape
        H = self.hidden_size
        h = h0 if h0 is not None else self._init_state(n)
        outs = []
        for t in range(T):
            xt = x[:, t, :]
            gi = xt @ self.W_ih + self.b
            gh = h @ self.W_hh
            r = (gi[:, :H] + gh[:, :H]).sigmoid()
            z = (gi[:, H:2 * H] + gh[:, H:2 * H]).sigmoid()
            nout = (gi[:, 2 * H:] + r * gh[:, 2 * H:]).tanh()
            h = (1.0 - z) * nout + z * h
            outs.append(h)
        return Tensor.stack(outs, axis=1), h


class LSTM(_RecurrentBase):
    def __init__(self, input_size, hidden_size, rng=None):
        super().__init__(input_size, hidden_size, n_gates=4, rng=rng)

    def forward(self, x, state=None):
        n, T, _ = x.shape
        H = self.hidden_size
        if state is None:
            h, c = self._init_state(n), self._init_state(n)
        else:
            h, c = state
        outs = []
        for t in range(T):
            xt = x[:, t, :]
            gates = xt @ self.W_ih + h @ self.W_hh + self.b
            i = gates[:, :H].sigmoid()
            f = gates[:, H:2 * H].sigmoid()
            g = gates[:, 2 * H:3 * H].tanh()
            o = gates[:, 3 * H:].sigmoid()
            c = f * c + i * g
            h = o * c.tanh()
            outs.append(h)
        return Tensor.stack(outs, axis=1), (h, c)


__all__ = ["RNN", "GRU", "LSTM"]
