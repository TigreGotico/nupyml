"""Dilated causal convolutions with GATED activations (van den Oord, 2016)."""
from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear
from .causal_conv1d import CausalConv1d


class WaveNet(Module):
    """Dilated causal convolutions with GATED activations (van den Oord, 2016).

    Built for raw audio: a stack of causal convolutions whose dilation DOUBLES each
    layer, so a modest stack sees thousands of past samples, and each layer uses a
    GATED activation ``tanh(conv_f(x)) * sigmoid(conv_g(x))`` -- the sigmoid gate
    decides how much of the tanh-filtered signal passes, an LSTM-style gate in
    convolutional form. Residual connections carry the input forward while SKIP
    connections from every layer are summed into the output, so gradients and
    features from all depths reach the end. Maps ``(batch, length, in_ch)`` to
    ``(batch, length, out_ch)``, strictly causal.
    """

    def __init__(self, in_ch, residual_ch=16, skip_ch=16, out_ch=1, n_layers=4,
                 kernel_size=2, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.start = Linear(in_ch, residual_ch, rng=r)
        self.filters, self.gates, self.res, self.skip = [], [], [], []
        for i in range(n_layers):
            d = 2 ** i
            self.filters.append(CausalConv1d(residual_ch, residual_ch,
                                             kernel_size, dilation=d, rng=r))
            self.gates.append(CausalConv1d(residual_ch, residual_ch,
                                           kernel_size, dilation=d, rng=r))
            self.res.append(Linear(residual_ch, residual_ch, rng=r))
            self.skip.append(Linear(residual_ch, skip_ch, rng=r))
        self.out1 = Linear(skip_ch, skip_ch, rng=r)
        self.out2 = Linear(skip_ch, out_ch, rng=r)

    def parameters(self):
        ps = list(self.start.parameters())
        for group in (self.filters, self.gates, self.res, self.skip):
            for m in group:
                ps += list(m.parameters())
        return ps + list(self.out1.parameters()) + list(self.out2.parameters())

    def forward(self, x):
        h = self.start(Tensor._wrap(x))
        skip_total = None
        for f, g, res, skip in zip(self.filters, self.gates, self.res, self.skip):
            gated = f(h).tanh() * g(h).sigmoid()        # gated activation unit
            skip_out = skip(gated)
            skip_total = skip_out if skip_total is None else skip_total + skip_out
            h = h + res(gated)                          # residual
        out = self.out2(self.out1(skip_total.relu()).relu())
        return out


__all__ = ["WaveNet"]
