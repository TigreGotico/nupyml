"""Attention in LINEAR time via a kernel feature map (Katharopoulos, 2020)."""
from ..autograd import Tensor
from .module import Module
from .layers import Linear, LayerNorm, ReLU, GELU


class LinearAttention(Module):
    """Attention in LINEAR time via a kernel feature map (Katharopoulos, 2020).

    Softmax attention costs ``O(n^2)`` because it forms the full n-by-n score
    matrix -- prohibitive for long sequences. Linear attention replaces
    ``softmax(QK^T)V`` with a feature map ``phi`` and reassociates::

        out = phi(Q) ( phi(K)^T V ) / ( phi(Q) phi(K)^T 1 )

    Computing ``phi(K)^T V`` first (a ``d-by-d`` matrix) makes the cost ``O(n d^2)``
    -- LINEAR in sequence length. The trade is a low-rank approximation to the
    attention matrix; here ``phi(x) = relu(x) + 1`` (a non-negative map).
    """

    def __init__(self, embed_dim, rng=None):
        super().__init__()
        self.q = Linear(embed_dim, embed_dim, rng=rng)
        self.k = Linear(embed_dim, embed_dim, rng=rng)
        self.v = Linear(embed_dim, embed_dim, rng=rng)

    def parameters(self):
        return list(self.q.parameters()) + list(self.k.parameters()) + list(self.v.parameters())

    def forward(self, x):
        x = Tensor._wrap(x)
        q = self.q(x).relu() + 1.0                     # phi(Q)  (b, n, d)
        k = self.k(x).relu() + 1.0                     # phi(K)
        v = self.v(x)
        kv = k.swapaxes(1, 2) @ v                       # phi(K)^T V  (b, d, d)
        num = q @ kv                                    # (b, n, d)
        # normaliser: phi(Q) . sum_n phi(K)
        z = q @ k.sum(axis=1).reshape(k.shape[0], k.shape[2], 1)   # (b, n, 1)
        return num / (z + 1e-6)


__all__ = ["LinearAttention"]
