"""Recurrent layers: carrying state along a sequence.

THE IDEA
--------
Process a sequence one step at a time, keeping a hidden state that summarises
everything seen so far::

    h_t = f(x_t, h_{t-1})

The SAME weights are applied at every step -- the recurrent analogue of a
convolution's weight sharing across space. So an RNN handles any sequence
length with a fixed parameter count.

THE VANISHING GRADIENT, AND WHY GATES EXIST
-------------------------------------------
Backpropagating through T steps multiplies by the recurrent matrix T times. Any
repeated multiplication is exponential: factors below 1 vanish, above 1 explode.
A plain ``RNN`` therefore cannot learn dependencies more than ~10 steps apart --
the gradient from step 100 reaches step 1 as approximately zero. Exploding
gradients are the loud failure and are fixed by ``clip_grad_norm``; vanishing
gradients are the silent one, and no clipping helps.

``LSTM`` and ``GRU`` solve it by making the state's default behaviour PERSIST
rather than transform. The LSTM's cell state is updated by ADDITION, not
multiplication by a weight matrix, so there is a path along which the gradient
flows unchanged -- exactly the trick a residual connection uses. Gates then
learn when to write to that path, when to erase it, and when to read from it.

WHEN TO USE THESE AT ALL
------------------------
Attention (``nupyml.nn.attention``) connects distant positions in one hop and
parallelises across the sequence, which is why transformers replaced RNNs for
most tasks. RNNs keep two advantages: cost linear rather than quadratic in
length, and O(1) state for streaming -- a transformer must re-read its whole
context.
"""
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
    """The plain recurrent layer: ``h = tanh(x @ W_ih + h @ W_hh + b)``.

    The simplest thing that could work, and instructive precisely because of how
    it fails. The state is REPLACED at every step by a matrix product squashed
    through tanh, so information must survive repeated multiplication to persist
    -- and tanh's derivative is at most 1, usually much less, which makes the
    decay through time essentially inevitable.

    Fine for short sequences. For anything longer, use ``GRU`` or ``LSTM``.
    """

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
    """Gated Recurrent Unit: the LSTM's idea with two gates instead of three.

    * **update gate** ``z`` -- how much of the old state to keep versus replace.
      This is the key one: the new state is ``(1-z)*candidate + z*h_old``, an
      interpolation. With ``z`` near 1 the state is copied through unchanged and
      the gradient passes intact, which is how long-range memory survives.
    * **reset gate** ``r`` -- how much of the old state the candidate is even
      allowed to see, letting the unit drop context that has become irrelevant.

    No separate cell state: the GRU keeps one vector where the LSTM keeps two,
    so it has fewer parameters and trains faster. In practice the two perform
    comparably, and which wins is task-dependent.
    """

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
    """Long Short-Term Memory: a protected cell state, controlled by gates.

    Two states are carried, and the split is the whole design:

    * ``c`` -- the CELL state, long-term memory. Updated only by
      ``c = f*c + i*g``: scaling and addition, never a matrix multiply. That
      makes it a near-uninterrupted highway along which gradients flow without
      the repeated multiplication that kills a plain RNN.
    * ``h`` -- the HIDDEN state, what the outside world sees, read out of ``c``
      through a gate.

    Three gates decide the traffic on that highway, each a sigmoid in [0, 1]:

    * **forget** ``f`` -- how much of the cell to erase. With ``f`` near 1 a
      memory persists indefinitely; near 0 it is wiped.
    * **input** ``i`` -- how much of the new candidate to write.
    * **output** ``o`` -- how much of the cell to expose as ``h``. The cell can
      hold something without acting on it yet.

    All four gate pre-activations are computed in ONE matmul and then sliced,
    which is why ``n_gates=4`` sizes the weights: one large matrix product is far
    faster than four small ones.
    """

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
