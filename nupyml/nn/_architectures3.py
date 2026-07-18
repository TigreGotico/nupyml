"""Neural architectures v3: gating, channel attention, causal convolution, graph
attention, and weight generation.

These extend the architecture set with structures for sequences (TCN), feature
recalibration (SE), very deep nets (Highway), graphs (GATv2), and networks that
generate other networks (hypernetworks).
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module, Parameter
from .layers import Linear, ReLU


class HighwayNetwork(Module):
    """A layer that LEARNS how much to transform vs pass through (Srivastava, 2015).

    Stacking many plain layers makes gradients vanish. A highway layer adds a
    learned GATE::

        y = T(x) * H(x) + (1 - T(x)) * x

    where ``H`` is a normal transform and ``T`` (a sigmoid "transform gate")
    decides, per unit, how much to transform versus carry the input straight
    through. When ``T=0`` the layer is the identity, so information (and gradient)
    flows unimpeded -- the idea that let networks go very deep, and the direct
    ancestor of the residual connection and the LSTM gate. Requires equal in/out
    dimension.
    """

    def __init__(self, dim, rng=None):
        super().__init__()
        self.H = Linear(dim, dim, rng=rng)
        self.T = Linear(dim, dim, rng=rng)
        # bias the gate toward CARRYING at init (negative gate bias) so deep stacks
        # start near the identity
        self.T.bias.data[...] = -4.0

    def parameters(self):
        return list(self.H.parameters()) + list(self.T.parameters())

    def forward(self, x):
        x = Tensor._wrap(x)
        t = self.T(x).sigmoid()
        return t * self.H(x).relu() + (1.0 - t) * x


class SqueezeExcitation(Module):
    """Channel attention: recalibrate features by their global importance
    (Hu et al., 2018).

    A network treats every feature channel equally, but some are far more useful
    for a given input. Squeeze-and-excitation SQUEEZES the feature vector to a
    per-channel summary, learns EXCITATION weights from it through a tiny
    bottleneck MLP (sigmoid-gated), and rescales each channel by its weight -- a
    cheap self-attention over channels that consistently improves accuracy. Here it
    gates a feature vector; in a CNN the squeeze is a global average pool.
    """

    def __init__(self, channels, reduction=4, rng=None):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = Linear(channels, hidden, rng=rng)
        self.fc2 = Linear(hidden, channels, rng=rng)

    def parameters(self):
        return list(self.fc1.parameters()) + list(self.fc2.parameters())

    def forward(self, x):
        x = Tensor._wrap(x)
        gate = self.fc2(self.fc1(x).relu()).sigmoid()   # per-channel importance
        return x * gate


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


class GATv2(Module):
    """Graph Attention Network v2 -- learn how much each neighbour matters
    (Brody et al., 2022).

    A GCN averages neighbours with fixed weights; GAT learns ATTENTION weights so a
    node listens more to relevant neighbours. GATv2 fixes the original GAT's flaw
    that its attention was "static" (the ranking of neighbours could not depend on
    the query node) by applying the nonlinearity BEFORE the attention vector, giving
    truly DYNAMIC attention. ``forward(X, A)`` takes node features ``(n, d)`` and an
    adjacency ``(n, n)`` and returns attention-aggregated features.
    """

    def __init__(self, in_dim, out_dim, rng=None):
        super().__init__()
        from ..utils import check_random_state
        r = check_random_state(rng)
        self.W = Linear(in_dim, out_dim, rng=r)
        self.a = Parameter(r.randn(2 * out_dim) * 0.1)   # attention vector

    def parameters(self):
        return list(self.W.parameters()) + [self.a]

    def forward(self, X, A):
        h = self.W(Tensor._wrap(X))                      # (n, out)
        n = h.shape[0]
        A = np.asarray(A)
        hd = h.data
        # GATv2 score: a . LeakyReLU(W h_i || W h_j)  -- nonlinearity before 'a'
        out_rows = []
        for i in range(n):
            nbrs = np.where(A[i] > 0)[0]
            if len(nbrs) == 0:
                out_rows.append(h[i:i + 1])
                continue
            scores = []
            for j in nbrs:
                cat = Tensor.concatenate([h[i:i + 1], h[j:j + 1]], axis=1)
                lr = cat.relu() + 0.01 * (cat - cat.relu())   # leaky relu
                scores.append((lr @ self.a.reshape(-1, 1)))
            s = Tensor.concatenate(scores, axis=0).reshape(1, -1)
            alpha = F.softmax(s, axis=1)                  # attention over neighbours
            agg = alpha @ h[nbrs]
            out_rows.append(agg)
        return Tensor.concatenate(out_rows, axis=0)


class HyperNetwork(Module):
    """A network that GENERATES another network's weights (Ha et al., 2017).

    Instead of learning a target layer's weights directly, learn a small
    HYPERNETWORK that OUTPUTS them from a context/embedding. This shares parameters
    across many target layers (one hypernet generates all of them), enables
    conditioning (different context -> different weights, e.g. per-task or
    per-timestep), and can compress a large model into a small generator. Here the
    hypernet maps a context vector to the flattened weight matrix of a linear
    target layer, which is then applied to the input.
    """

    def __init__(self, context_dim, in_dim, out_dim, hidden=16, rng=None):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.gen = Linear(context_dim, in_dim * out_dim, rng=rng)
        self.ctx_hidden = Linear(context_dim, hidden, rng=rng)

    def parameters(self):
        return list(self.gen.parameters())

    def generate_weight(self, context):
        w = self.gen(Tensor._wrap(context))              # (batch, in*out)
        return w.reshape(-1, self.in_dim, self.out_dim)

    def forward(self, context, x):
        W = self.generate_weight(context)                # per-sample weight matrix
        x = Tensor._wrap(x)
        # apply each sample's generated weight to its input
        return (x.reshape(x.shape[0], 1, self.in_dim) @ W).reshape(x.shape[0], self.out_dim)


__all__ = ["HighwayNetwork", "SqueezeExcitation", "CausalConv1d",
           "TemporalConvNet", "GATv2", "HyperNetwork"]
