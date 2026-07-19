"""Let the network WARP its own input before reading it (Jaderberg, 2015)."""
import numpy as np
from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear


def _bilinear_sample(img, sx, sy):
    H, W = img.shape
    x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    wx = sx - x0; wy = sy - y0
    def gv(yy, xx):
        inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        v = np.zeros_like(sx)
        yy = np.clip(yy, 0, H - 1); xx = np.clip(xx, 0, W - 1)
        v[inb] = img[yy[inb], xx[inb]]
        return v
    return (gv(y0, x0) * (1 - wx) * (1 - wy) + gv(y0, x1) * wx * (1 - wy)
            + gv(y1, x0) * (1 - wx) * wy + gv(y1, x1) * wx * wy)


class SpatialTransformer(Module):
    """Let the network WARP its own input before reading it (Jaderberg, 2015).

    CNNs are only locally translation-invariant; they struggle when the object is
    rotated, scaled, or shifted. A spatial transformer inserts a small module that
    LOOKS at the input, predicts an affine transform ``theta`` (a localisation
    net), builds the corresponding sampling GRID, and bilinearly SAMPLES the input
    onto it -- actively de-rotating or centring the object so the downstream net
    sees a canonical view. The whole warp is differentiable through the bilinear
    sampler. Here ``localize`` predicts theta and ``transform`` applies the affine
    grid sample to a batch of ``(H, W)`` images.
    """

    def __init__(self, in_h, in_w, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.in_h, self.in_w = in_h, in_w
        self.loc = Linear(in_h * in_w, 6, rng=r)
        # bias the localisation to the IDENTITY transform at init
        self.loc.weight.data[...] *= 0.01
        self.loc.bias.data[...] = np.array([1., 0., 0., 0., 1., 0.])

    def parameters(self):
        return list(self.loc.parameters())

    def localize(self, images):
        flat = Tensor(np.asarray(images).reshape(len(images), -1))
        return self.loc(flat).data.reshape(-1, 2, 3)

    def transform(self, images, theta=None):
        """Affine-warp each image via a sampling grid + bilinear interpolation."""
        images = np.asarray(images, dtype=float)
        N, H, W = images.shape
        if theta is None:
            theta = self.localize(images)
        # normalised output grid in [-1, 1]
        ys, xs = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W),
                             indexing="ij")
        grid = np.stack([xs.ravel(), ys.ravel(), np.ones(H * W)], axis=0)  # (3, HW)
        out = np.empty((N, H, W))
        for n in range(N):
            src = theta[n] @ grid                       # (2, HW) source coords in [-1,1]
            sx = (src[0] + 1) * (W - 1) / 2             # to pixel coords
            sy = (src[1] + 1) * (H - 1) / 2
            out[n] = _bilinear_sample(images[n], sx, sy).reshape(H, W)
        return out


__all__ = ["SpatialTransformer"]
