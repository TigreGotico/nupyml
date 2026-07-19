"""Optimise for perceived STRUCTURE, not per-pixel error (Wang et al., 2004)."""
from ..autograd import Tensor
from .module import Module


class SSIMLoss(Module):
    """Optimise for perceived STRUCTURE, not per-pixel error (Wang et al., 2004).

    Two images can have identical MSE while one looks fine and the other is
    visibly wrong, because MSE is blind to structure. The structural similarity
    index compares LUMINANCE, CONTRAST and STRUCTURE (via local means, variances
    and covariance), matching human judgement far better -- which is why it is the
    standard image-quality metric and a common training loss for super-resolution
    and denoising. ``1 - SSIM`` so that identical images give zero loss. Uses
    global per-image statistics.
    """

    def __init__(self, c1=0.01 ** 2, c2=0.03 ** 2):
        super().__init__()
        self.c1, self.c2 = c1, c2

    def forward(self, pred, target):
        x = Tensor._wrap(pred); y = Tensor._wrap(target)
        mx = x.mean(); my = y.mean()
        vx = ((x - mx) * (x - mx)).mean()
        vy = ((y - my) * (y - my)).mean()
        cov = ((x - mx) * (y - my)).mean()
        ssim = ((2 * mx * my + self.c1) * (2 * cov + self.c2)) / \
               ((mx * mx + my * my + self.c1) * (vx + vy + self.c2))
        return 1.0 - ssim


__all__ = ["SSIMLoss"]
