"""One decoder-only (GPT-style) block: causal self-attention + feed-forward."""
from .module import Module
from .layers import Linear, LayerNorm, Embedding, Dropout, ReLU, GELU
from .attention import MultiHeadAttention, causal_mask


class TransformerDecoderLayer(Module):
    """One decoder-only (GPT-style) block: causal self-attention + feed-forward.

    Unlike the encoder layer, the self-attention is MASKED so position ``t`` can
    only attend to positions ``<= t`` -- the model must predict each token from
    the past alone, never peeking ahead. Pre-norm residual around both the
    attention and the feed-forward sublayers.
    """

    def __init__(self, embed_dim, num_heads, ff_dim=None, rng=None):
        super().__init__()
        ff_dim = ff_dim or 4 * embed_dim
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, rng=rng)
        self.norm2 = LayerNorm(embed_dim)
        self.fc1 = Linear(embed_dim, ff_dim, rng=rng)
        self.act = GELU()
        self.fc2 = Linear(ff_dim, embed_dim, rng=rng)

    def forward(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, mask=mask)         # masked self-attention
        h = self.norm2(x)
        return x + self.fc2(self.act(self.fc1(h)))    # feed-forward


__all__ = ["TransformerDecoderLayer"]
