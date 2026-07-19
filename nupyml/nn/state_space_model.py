"""A trainable linear state-space (diagonal SSM) for long sequences"""
import numpy as np
from ..autograd import Tensor
from .module import Module
from .layers import Linear, LayerNorm, ReLU, GELU


class StateSpaceModel(Module):
    """A trainable linear state-space (diagonal SSM) for long sequences
    (the S4/Mamba idea).

    An RNN mixes memory and nonlinearity and is slow and forgetful over long
    ranges. A linear STATE-SPACE model keeps a linear recurrence
    ``x_t = A x_{t-1} + B u_t``, ``y_t = C x_t`` with a DIAGONAL ``A`` -- so each
    hidden dimension is an independent leaky integrator with its own decay rate.
    Diagonal ``A`` makes it cheap and, crucially, lets you set very slow decays
    that carry information across thousands of steps -- the long-range memory RNNs
    lack. ``A`` is parameterised as ``-softplus`` (via sigmoid gate here) to stay
    stable (decays in (0, 1)). Input ``(batch, T, in_dim)`` -> ``(batch, T, out_dim)``.
    """

    def __init__(self, in_dim, state_dim=16, out_dim=1, rng=None):
        super().__init__()
        self.encode = Linear(in_dim, state_dim, rng=rng)
        self.decode = Linear(state_dim, out_dim, rng=rng)
        # log-decay -> A = sigmoid(a) in (0,1), a stable per-dim decay
        self.a = Linear(1, state_dim, rng=rng)
        self.state_dim = state_dim

    def parameters(self):
        return (list(self.encode.parameters()) + list(self.decode.parameters())
                + list(self.a.parameters()))

    def forward(self, u):
        u = Tensor._wrap(u)
        b, T, _ = u.shape
        Bu = self.encode(u)                            # (b, T, state) = B u_t
        decay = self.a(Tensor(np.ones((1, 1)))).sigmoid()   # (1, state), in (0,1)
        outs = []
        x = Tensor(np.zeros((b, self.state_dim)))
        for t in range(T):
            x = x * decay + Bu[:, t]                    # x_t = A x_{t-1} + B u_t
            outs.append(self.decode(x))
        return Tensor.stack(outs, axis=1)              # (b, T, out_dim)


__all__ = ["StateSpaceModel"]
