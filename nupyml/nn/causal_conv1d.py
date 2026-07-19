"""A dilated CAUSAL 1-D convolution: output t depends only on inputs <= t."""
import numpy as np
from ..autograd import Tensor
from .module import Module, Parameter


class CausalConv1d(Module):
    """A dilated CAUSAL 1-D convolution: output t depends only on inputs <= t."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, rng=None):
        super().__init__()
        from ..utils import check_random_state
        r = check_random_state(rng)
        self.k = kernel_size
        self.dilation = dilation
        self.W = Parameter(r.randn(kernel_size, in_ch, out_ch) * 0.1)
        self.b = Parameter(np.zeros(out_ch))

    def parameters(self):
        return [self.W, self.b]

    def forward(self, x):
        # x: (batch, length, in_ch); left-pad so the conv cannot see the future
        x = Tensor._wrap(x)
        pad = (self.k - 1) * self.dilation
        B, Ln, C = x.shape
        xp = Tensor.concatenate([Tensor(np.zeros((B, pad, C))), x], axis=1)
        out = self.b
        for i in range(self.k):                          # sum the dilated taps
            sl = xp[:, i * self.dilation: i * self.dilation + Ln, :]
            out = out + sl @ self.W[i]
        return out


__all__ = ["CausalConv1d"]
