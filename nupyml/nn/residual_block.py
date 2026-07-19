"""A pre-norm residual MLP block: ``x + FF(norm(x))``."""
from .module import Module
from .layers import Linear, LayerNorm, Embedding, Dropout, ReLU, GELU


class ResidualBlock(Module):
    """A pre-norm residual MLP block: ``x + FF(norm(x))``.

    THE POINT OF THE SKIP
    ---------------------
    Stacking many layers makes gradients vanish and the identity map hard to
    learn. A residual block computes a small CORRECTION and ADDS it to the input,
    so the default behaviour is to pass the input through unchanged and each block
    only has to learn the delta. That is what let networks go from tens of layers
    to hundreds -- gradients flow straight down the skip connections. Pre-norm
    (LayerNorm before the sublayer) is the arrangement that trains deep stacks
    stably.
    """

    def __init__(self, dim, hidden=None, rng=None):
        super().__init__()
        hidden = hidden or 4 * dim
        self.norm = LayerNorm(dim)
        self.fc1 = Linear(dim, hidden, rng=rng)
        self.act = GELU()
        self.fc2 = Linear(hidden, dim, rng=rng)

    def forward(self, x):
        h = self.fc2(self.act(self.fc1(self.norm(x))))
        return x + h                                  # the residual connection


__all__ = ["ResidualBlock"]
