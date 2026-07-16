"""Attention and transformer building blocks."""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from ..utils import check_random_state
from .module import Module
from .layers import Linear, LayerNorm, Dropout, GELU


def scaled_dot_product_attention(q, k, v, mask=None):
    """q,k,v: (..., T, D). mask: broadcastable bool array, True = keep."""
    d = q.shape[-1]
    scores = (q @ k.swapaxes(-1, -2)) * (1.0 / np.sqrt(d))
    if mask is not None:
        neg = np.where(mask, 0.0, -1e9)
        scores = scores + Tensor(neg)
    attn = F.softmax(scores, axis=-1)
    return attn @ v, attn


class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads, rng=None):
        super().__init__()
        assert embed_dim % num_heads == 0
        rng = check_random_state(rng)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, rng=rng)
        self.k_proj = Linear(embed_dim, embed_dim, rng=rng)
        self.v_proj = Linear(embed_dim, embed_dim, rng=rng)
        self.out_proj = Linear(embed_dim, embed_dim, rng=rng)

    def _split(self, x):
        n, t, _ = x.shape
        return x.reshape(n, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, query, key=None, value=None, mask=None):
        key = query if key is None else key
        value = key if value is None else value
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))
        out, attn = scaled_dot_product_attention(q, k, v, mask=mask)
        n, h, t, hd = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(n, t, h * hd)
        self.last_attn_ = attn.data
        return self.out_proj(out)


def causal_mask(T):
    return np.tril(np.ones((T, T), dtype=bool))


class TransformerEncoderLayer(Module):
    def __init__(self, embed_dim, num_heads, ff_dim=None, dropout=0.0, rng=None):
        super().__init__()
        rng = check_random_state(rng)
        ff_dim = ff_dim or 4 * embed_dim
        self.attn = MultiHeadAttention(embed_dim, num_heads, rng=rng)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.ff1 = Linear(embed_dim, ff_dim, rng=rng)
        self.ff2 = Linear(ff_dim, embed_dim, rng=rng)
        self.act = GELU()
        self.drop = Dropout(dropout, rng=rng)

    def forward(self, x, mask=None):
        # pre-norm residual blocks
        x = x + self.drop(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop(self.ff2(self.act(self.ff1(self.norm2(x)))))
        return x


class PositionalEncoding(Module):
    """Fixed sinusoidal positional encoding added to (N, T, D) input."""

    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pos = np.arange(max_len)[:, None]
        i = np.arange(0, embed_dim, 2)[None, :]
        angle = pos / np.power(10000.0, i / embed_dim)
        pe = np.zeros((max_len, embed_dim))
        pe[:, 0::2] = np.sin(angle)
        pe[:, 1::2] = np.cos(angle)[:, : pe[:, 1::2].shape[1]]
        self.pe = pe

    def forward(self, x):
        T = x.shape[1]
        return x + Tensor(self.pe[None, :T, :])


__all__ = ["scaled_dot_product_attention", "MultiHeadAttention", "causal_mask",
           "TransformerEncoderLayer", "PositionalEncoding"]
