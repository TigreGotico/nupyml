"""A stack of dilated causal convolutions with residuals (Bai et al., 2018)."""
from ..autograd import Tensor
from .module import Module, Parameter
from .layers import Linear, ReLU
from .causal_conv1d import CausalConv1d


class TemporalConvNet(Module):
    """A stack of dilated causal convolutions with residuals (Bai et al., 2018).

    RNNs process sequences step-by-step (slow, forgetful). A TCN uses CONVOLUTIONS
    with EXPONENTIALLY GROWING DILATION, so a stack of ``L`` layers sees a receptive
    field of ``O(2^L)`` timesteps in parallel, and CAUSAL padding guarantees no
    leakage from the future. Residual connections keep it trainable when deep. On
    many sequence tasks it matches or beats LSTMs while training far faster (all
    positions computed at once). ``forward`` maps ``(batch, length, in_ch)`` to
    ``(batch, length, channels)``.
    """

    def __init__(self, in_ch, channels=16, n_layers=3, kernel_size=2, rng=None):
        super().__init__()
        self.blocks = []
        c = in_ch
        for i in range(n_layers):
            conv = CausalConv1d(c, channels, kernel_size, dilation=2 ** i, rng=rng)
            res = Linear(c, channels, rng=rng) if c != channels else None
            self.blocks.append((conv, res))
            c = channels

    def parameters(self):
        ps = []
        for conv, res in self.blocks:
            ps += conv.parameters()
            if res is not None:
                ps += list(res.parameters())
        return ps

    def forward(self, x):
        h = Tensor._wrap(x)
        for conv, res in self.blocks:
            out = conv(h).relu()
            skip = h if res is None else res(h)          # residual (matched dims)
            h = out + skip
        return h


__all__ = ["TemporalConvNet"]
