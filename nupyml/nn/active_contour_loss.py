"""Active-contour loss: a boundary-aware segmentation loss (length + region)."""
import numpy as np

from ..autograd import Tensor
from .module import Module


class ActiveContourLoss(Module):
    """Penalise the BOUNDARY, not just per-pixel overlap (Chen et al., 2019).

    Dice and cross-entropy score each pixel independently, so a ragged or leaky
    boundary can still score well. The active-contour loss borrows the classic
    snakes energy: a LENGTH term (the integral of the prediction's gradient
    magnitude) that punishes jagged, over-long contours, plus a REGION term that
    wants the inside and outside of the predicted mask to match the target's two
    regions. Minimising it yields smooth, well-placed boundaries -- exactly what
    per-pixel losses neglect. ``weight`` trades length against region. Predictions
    are probability maps in [0, 1]; targets are binary masks.
    """

    def __init__(self, weight=1.0, eps=1e-8):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, pred, target):
        pred = Tensor._wrap(pred)
        y = np.asarray(target if not isinstance(target, Tensor) else target.data,
                       dtype=float)
        # length term: total variation of the prediction (contour length)
        dx = pred[:, 1:, :] - pred[:, :-1, :]
        dy = pred[:, :, 1:] - pred[:, :, :-1]
        length = ((dx * dx)[:, :, :-1] + (dy * dy)[:, :-1, :] + self.eps).sqrt().sum()
        # region term: inside should be target=1, outside should be target=0
        inside = (pred * Tensor((y - 1.0) ** 2)).sum()
        outside = ((1.0 - pred) * Tensor((y - 0.0) ** 2)).sum()
        return self.weight * length + (inside + outside)


__all__ = ["ActiveContourLoss"]
