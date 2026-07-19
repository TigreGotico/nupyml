"""Attention scored by the DISTANCE between tokens (Shaw 2018; Dai 2019)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear


class RelativePositionAttention(Module):
    """Attention scored by the DISTANCE between tokens (Shaw 2018; Dai 2019).

    Absolute positional encodings tie a token's meaning to WHERE it sits, so a
    pattern learned at position 5 does not transfer to position 50. Relative
    attention instead adds a learned bias that depends only on the OFFSET ``i - j``
    between query and key, so the same relative pattern is recognised anywhere in
    the sequence (and it extrapolates to longer sequences). This is the mechanism
    behind Transformer-XL. Single-head here for clarity; input ``(batch, seq, dim)``.
    """

    def __init__(self, dim, max_len=64, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.dim = dim
        self.q = Linear(dim, dim, rng=r)
        self.k = Linear(dim, dim, rng=r)
        self.v = Linear(dim, dim, rng=r)
        self.max_len = max_len
        # a learned bias per relative offset in [-(L-1), L-1]
        self.rel_bias = Parameter(r.randn(2 * max_len - 1) * 0.02)

    def parameters(self):
        return (list(self.q.parameters()) + list(self.k.parameters())
                + list(self.v.parameters()) + [self.rel_bias])

    def _bias_matrix(self, n):
        idx = np.arange(n)
        offset = idx[:, None] - idx[None, :] + (self.max_len - 1)   # map to [0, 2L-2]
        return offset

    def forward(self, x):
        x = Tensor._wrap(x)
        B, n, d = x.shape
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores = (Q @ K.transpose(0, 2, 1)) * (1.0 / np.sqrt(d))    # (B, n, n)
        bias = self.rel_bias[self._bias_matrix(n).ravel()].reshape(1, n, n)
        attn = F.softmax(scores + bias, axis=2)                     # relative bias
        return attn @ V


__all__ = ["RelativePositionAttention"]
