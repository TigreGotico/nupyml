"""A plain transformer over image PATCHES (Dosovitskiy et al., 2021)."""
import numpy as np
from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear
from .attention import TransformerEncoderLayer


class VisionTransformer(Module):
    """A plain transformer over image PATCHES (Dosovitskiy et al., 2021).

    Convolutions bake in locality and translation equivariance. The Vision
    Transformer throws that away: cut the image into a grid of patches, linearly
    embed each patch as a "token", prepend a learnable CLASS token, add positional
    embeddings, and run an ordinary transformer encoder -- then classify from the
    class token's output. With enough data it matches or beats CNNs, showing the
    convolutional prior is helpful but not necessary. Input images are
    ``(batch, H, W)`` (single channel); ``patch_size`` must divide ``H`` and ``W``.
    """

    def __init__(self, img_size, patch_size, n_classes, embed_dim=32, depth=2,
                 n_heads=4, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.p = patch_size
        self.grid = img_size // patch_size
        self.n_patches = self.grid * self.grid
        self.patch_embed = Linear(patch_size * patch_size, embed_dim, rng=r)
        self.cls_token = Parameter(r.randn(1, embed_dim) * 0.02)
        self.pos_embed = Parameter(r.randn(self.n_patches + 1, embed_dim) * 0.02)
        self.layers = [TransformerEncoderLayer(embed_dim, n_heads, rng=r)
                       for _ in range(depth)]
        self.head = Linear(embed_dim, n_classes, rng=r)

    def parameters(self):
        ps = list(self.patch_embed.parameters()) + [self.cls_token, self.pos_embed]
        for l in self.layers:
            ps += list(l.parameters())
        return ps + list(self.head.parameters())

    def _patchify(self, images):
        images = np.asarray(images, float)
        B = len(images); p, g = self.p, self.grid
        out = np.empty((B, self.n_patches, p * p))
        for i in range(g):
            for j in range(g):
                patch = images[:, i * p:(i + 1) * p, j * p:(j + 1) * p]
                out[:, i * g + j] = patch.reshape(B, -1)
        return out

    def forward(self, images):
        B = len(images)
        tokens = self.patch_embed(Tensor(self._patchify(images)))   # (B, n, d)
        cls = self.cls_token.reshape(1, 1, -1)
        cls = Tensor(np.tile(cls.data, (B, 1, 1)))
        x = Tensor.concatenate([cls, tokens], axis=1) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.head(x[:, 0, :])                                 # class token


__all__ = ["VisionTransformer"]
