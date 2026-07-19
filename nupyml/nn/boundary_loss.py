"""Weight errors by their distance to the true BOUNDARY (Kervadec et al., 2019)."""
import numpy as np
import scipy.ndimage as ndi
from ..autograd import Tensor
from .module import Module


class BoundaryLoss(Module):
    """Weight errors by their distance to the true BOUNDARY (Kervadec et al., 2019).

    Region losses (Dice, cross-entropy) treat every pixel equally, so a thin or
    small structure -- a vessel, a lesion edge -- contributes almost nothing and is
    ignored. The boundary loss instead integrates the prediction against a SIGNED
    DISTANCE map of the ground-truth boundary: predicting foreground far OUTSIDE the
    object (large positive distance) is penalised heavily, predicting inside is
    rewarded, and the emphasis concentrates exactly at the contour. Combined with a
    region loss it markedly improves boundary accuracy on imbalanced segmentation.
    ``target`` is a binary mask.
    """

    def __init__(self):
        super().__init__()

    def _signed_distance(self, mask):
        mask = np.asarray(mask).astype(bool)
        if mask.all() or not mask.any():
            return np.zeros(mask.shape, float)
        out = ndi.distance_transform_edt(~mask)            # distance outside (+)
        inside = ndi.distance_transform_edt(mask)          # distance inside (-)
        return out - inside

    def forward(self, pred, target):
        pred = Tensor._wrap(pred)
        phi = self._signed_distance(target)                # precomputed distance map
        return (pred * Tensor(phi)).mean()                 # integral of pred·distance


__all__ = ["BoundaryLoss"]
