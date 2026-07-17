"""Positional encoding for attention: telling a permutation-blind model about order.

THE PROBLEM ATTENTION HAS
-------------------------
Self-attention is PERMUTATION-EQUIVARIANT: shuffle the input tokens and the
outputs shuffle the same way, but nothing else changes. It has no built-in notion
of order -- "dog bites man" and "man bites dog" look identical to raw attention.
Something must inject position, and how you do it turns out to matter a great
deal for how well the model handles sequences longer than it trained on.

THE PROGRESSION
---------------
* Sinusoidal / learned absolute (in ``attention.py``) -- ADD a position vector to
  each token. Simple, but tied to absolute positions, so it generalises poorly
  past the trained length.
* ``RotaryPositionalEmbedding`` (RoPE) -- ROTATE the query and key vectors by an
  angle proportional to position. The dot product between two tokens then depends
  only on their RELATIVE offset -- exactly what attention should care about.
* ``ALiBi`` -- add a distance-proportional PENALTY to attention scores. No
  position vectors at all, and it extrapolates to far longer sequences than it
  trained on.

Both RoPE and ALiBi encode RELATIVE position, and that is why they replaced
absolute encodings in modern LLMs -- what matters is how far apart two tokens are,
not where they sit from the start.
"""
import numpy as np

from ..autograd import Tensor


class RotaryPositionalEmbedding:
    """RoPE: rotate q and k so their dot product depends on relative position.

    THE TRICK
    ---------
    Split each vector into 2D pairs and ROTATE each pair by an angle proportional
    to the token's position (different frequencies for different pairs, as in the
    sinusoidal encoding). The key identity: after rotating query at position ``m``
    and key at position ``n``, their dot product depends only on ``m - n`` -- the
    absolute positions cancel, leaving pure RELATIVE position.

    Why that is the right thing: attention scores a query against a key by their
    dot product, and what should matter is how far apart the two tokens are, not
    their absolute indices. RoPE builds that directly into the geometry, adds no
    parameters, and -- because it is a rotation applied at use time -- extends to
    positions never seen in training far more gracefully than a learned absolute
    table. It is the position encoding in LLaMA, GPT-NeoX, and most current models.

    Su et al. (2021).
    """

    def __init__(self, dim, base=10000.0):
        if dim % 2 != 0:
            raise ValueError("RoPE needs an even head dimension")
        self.dim = dim
        # one frequency per 2D pair, geometrically spaced -- fast-rotating pairs
        # capture short-range offsets, slow ones long-range
        self.inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))

    def _angles(self, seq_len):
        pos = np.arange(seq_len)
        # outer product: angle for every (position, frequency) pair
        return np.outer(pos, self.inv_freq)         # (seq_len, dim/2)

    def rotate(self, x):
        """Apply the rotation to a (..., seq_len, dim) array of q or k."""
        x = np.asarray(x.data if isinstance(x, Tensor) else x)
        seq_len = x.shape[-2]
        angles = self._angles(seq_len)
        cos = np.cos(angles)
        sin = np.sin(angles)
        # broadcast cos/sin over any leading (batch, head) axes
        for _ in range(x.ndim - 2):
            cos = cos[None]
            sin = sin[None]
        # split into the even/odd halves of each pair and apply the 2D rotation
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        out = np.empty_like(x)
        out[..., 0::2] = rot1
        out[..., 1::2] = rot2
        return out


class ALiBi:
    """Attention with Linear Biases: penalise distant tokens, no position vectors.

    THE IDEA
    --------
    Add nothing to the tokens. Instead, subtract from each attention SCORE a
    penalty proportional to how far apart the two tokens are::

        score(i, j) += -slope * |i - j|

    Nearer tokens are favoured, farther ones discounted -- a soft, built-in
    recency bias. Each attention head gets a DIFFERENT slope (a geometric
    sequence), so some heads look narrowly local and others range far, covering
    a spread of distances.

    THE PAYOFF: EXTRAPOLATION
    -------------------------
    Because the bias is just a function of distance -- not a learned table or a
    rotation with a trained range -- it is DEFINED for any distance, including ones
    longer than training. ALiBi-trained models run on sequences several times
    longer than they ever saw, with little degradation, which absolute and even
    rotary encodings struggle to do. Simplicity buying generalisation.

    Press, Smith & Lewis (2021).
    """

    def __init__(self, n_heads):
        self.n_heads = n_heads
        self.slopes = self._get_slopes(n_heads)

    @staticmethod
    def _get_slopes(n):
        """Geometric slopes, so heads cover a spread of distance scales.

        The published choice is powers of ``2^(-8/n)`` -- steep-sloped heads see
        only nearby tokens, shallow ones see far. The exact geometry is what makes
        a stack of heads jointly cover short and long range.
        """
        start = 2 ** (-8 / n) if n > 0 else 1.0
        return np.array([start ** (i + 1) for i in range(n)])

    def bias(self, seq_len):
        """The (n_heads, seq_len, seq_len) additive bias, added to raw scores.

        Zero on the diagonal (a token vs itself), growing negative with distance.
        For causal attention only the lower triangle is used.
        """
        pos = np.arange(seq_len)
        # |i - j|, the distance between every pair of positions
        distance = -np.abs(pos[:, None] - pos[None, :])
        return self.slopes[:, None, None] * distance[None, :, :]


__all__ = ["RotaryPositionalEmbedding", "ALiBi"]
