"""Attention: letting every position look directly at every other.

THE PROBLEM IT SOLVES
---------------------
An RNN carries information along a chain, one step at a time, so relating two
tokens 500 apart means routing a signal through 500 sequential updates -- during
which it is repeatedly multiplied by the same matrix and generally lost. And
because each step depends on the previous one, none of it parallelises.

Attention connects every position to every other DIRECTLY, in one step. Any two
tokens are one hop apart regardless of distance, and every position is computed
simultaneously. That is the entire reason transformers displaced RNNs: not
merely better accuracy, but that they train in parallel.

The cost is quadratic: n positions each attending to n positions. RNNs are
linear in sequence length. That quadratic term is what all the "efficient
attention" literature exists to attack.

THE MECHANISM
-------------
A soft dictionary lookup. Each position emits a QUERY (what am I looking for?),
a KEY (what do I offer?) and a VALUE (what do I pass on if chosen?). Every query
is compared against every key, the scores are softmaxed into weights, and the
output is the weighted sum of values. A hard lookup would take one value;
attention takes a blend.

Since every position is treated identically and the sum is order-independent,
attention has no idea what ORDER the tokens are in -- which is why
``PositionalEncoding`` exists.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from ..utils import check_random_state
from .module import Module
from .layers import Linear, LayerNorm, Dropout, GELU


def scaled_dot_product_attention(q, k, v, mask=None):
    """``softmax(Q K' / sqrt(d)) V``.

    WHY DIVIDE BY sqrt(d)
    ---------------------
    The dot product of two independent random d-dimensional vectors has variance
    proportional to d. So for large d the raw scores are spread very wide, and
    softmax of a wide range is nearly one-hot -- attention collapses onto a
    single position and the gradient through softmax goes to almost zero.
    Training stalls, and worse for larger models.

    Dividing by ``sqrt(d)`` normalises the variance back to ~1, keeping the
    softmax in a responsive range. It is the "scaled" in scaled dot-product
    attention, and it is what makes the mechanism work at scale.

    THE MASK
    --------
    Masked positions get ``-1e9`` added BEFORE the softmax, so ``exp`` sends
    them to ~0 and the remaining weights still sum to 1. Adding after the softmax
    would break the normalisation. ``-inf`` would be exact but produces ``nan``
    if a row is fully masked.
    """
    d = q.shape[-1]
    scores = (q @ k.swapaxes(-1, -2)) * (1.0 / np.sqrt(d))
    if mask is not None:
        neg = np.where(mask, 0.0, -1e9)
        scores = scores + Tensor(neg)
    attn = F.softmax(scores, axis=-1)
    return attn @ v, attn


class MultiHeadAttention(Module):
    """Run several attention operations in parallel over slices of the features.

    One attention head produces one set of weights, so it can express one
    relationship per position -- and a softmax mostly picks ONE place to look.
    But a word may need its subject, its adjective and its clause boundary at
    once.

    So split the features into ``num_heads`` groups and attend independently
    within each, then concatenate. Each head is free to specialise -- one
    tracking syntax, another position, another coreference -- and the output
    projection mixes their findings.

    Note this is nearly free: heads split the existing dimension rather than
    adding to it, so ``num_heads`` costs almost nothing beyond the same matmuls
    reshaped.
    """

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
    """Lower-triangular mask: position i may see only positions <= i.

    Required for generative models. Predicting token i+1 while being allowed to
    look at token i+1 is trivial and teaches nothing -- the model would score
    perfectly in training and produce nonsense when generating, where the future
    genuinely does not exist.

    Masking makes every position simultaneously a training example with the
    correct context, which is why a decoder trains on a whole sequence in one
    parallel pass rather than one token at a time.
    """
    return np.tril(np.ones((T, T), dtype=bool))


class TransformerEncoderLayer(Module):
    """Attention, then a feed-forward network, each wrapped in a residual.

    The two sublayers do different jobs: attention MIXES information between
    positions, and the feed-forward network then processes each position
    independently (which is why it is wide -- typically 4x -- since it is where
    most of the parameters and most of the per-token computation live).

    THE RESIDUALS
    -------------
    ``x + sublayer(x)``, not ``sublayer(x)``. The addition gives the gradient an
    unobstructed path back to earlier layers: differentiating a sum passes the
    gradient through unchanged, so it cannot vanish through depth no matter how
    many layers are stacked. Residual connections are what make very deep
    networks trainable at all.

    PRE-NORM
    --------
    This normalises the INPUT to each sublayer (``x + f(norm(x))``) rather than
    the output (``norm(x + f(x))``). Pre-norm keeps the residual path completely
    clean -- an unmodified straight line from input to output -- which trains
    stably without the learning-rate warmup that post-norm requires.
    """

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
    """Add position information, since attention is order-blind.

    Attention sums over positions, and a sum does not care about order --
    shuffling the tokens would give the same result. So position must be
    injected into the representations themselves.

    These are fixed sinusoids of geometrically increasing wavelength, added to
    the embeddings. Two useful properties:

    * Multiple frequencies encode position at multiple scales, much like the
      digits of a binary number, so nearby positions have similar codes and
      distant ones do not.
    * Sinusoids make RELATIVE position linearly recoverable: shifting by a fixed
      offset k is a rotation, the same rotation regardless of absolute position.
      So "three tokens back" is a consistent relationship the model can learn
      once.

    Being fixed rather than learned, it also extends to sequences longer than
    anything seen in training.
    """

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
