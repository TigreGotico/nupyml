"""Attention-based set model with pooling-by-attention (Lee et al., 2019)."""
import numpy as np
from ..autograd import Tensor
from .module import Module
from .layers import Linear, LayerNorm, ReLU, GELU
from .attention import MultiHeadAttention
from .deep_sets import DeepSets


class SetTransformer(Module):
    """Attention-based set model with pooling-by-attention (Lee et al., 2019).

    DeepSets pools with a plain sum, so it cannot model INTERACTIONS between set
    elements ("is there a pair that...?"). The Set Transformer replaces both the
    per-element map and the pooling with ATTENTION: a self-attention block lets
    elements attend to each other, and a learnable SEED vector attends to the set
    to pool it (pooling-by-multihead-attention, PMA). More expressive than
    DeepSets, still permutation-invariant. Input ``(batch, set_size, features)``.
    """

    def __init__(self, in_dim, embed_dim=32, num_heads=4, out_dim=1, rng=None):
        super().__init__()
        self.proj = Linear(in_dim, embed_dim, rng=rng)
        self.sab = MultiHeadAttention(embed_dim, num_heads, rng=rng)   # self-attn
        self.norm = LayerNorm(embed_dim)
        self.seed = Linear(1, embed_dim, rng=rng)     # produces the pooling query
        self.pma = MultiHeadAttention(embed_dim, num_heads, rng=rng)
        self.out = Linear(embed_dim, out_dim, rng=rng)

    def parameters(self):
        return (list(self.proj.parameters()) + list(self.sab.parameters())
                + list(self.norm.parameters()) + list(self.seed.parameters())
                + list(self.pma.parameters()) + list(self.out.parameters()))

    def forward(self, x):
        x = Tensor._wrap(x)
        b = x.shape[0]
        z = self.proj(x)
        z = self.norm(z + self.sab(z, z, z))          # elements attend to each other
        seed = self.seed(Tensor(np.ones((b, 1, 1))))  # one learnable seed per batch
        pooled = self.pma(seed, z, z)                 # seed attends to the set
        return self.out(pooled[:, 0])


__all__ = ["SetTransformer"]
