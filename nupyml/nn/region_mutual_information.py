"""Tie a segmentation to its target through MUTUAL INFORMATION (Zhao et al., 2019)."""
import numpy as np
from ..autograd import Tensor


def _mi_term(pjoint, pa, pb):
    return pjoint * (pjoint / (pa * pb + 1e-9) + 1e-9).log()


def region_mutual_information(pred, target):
    """Tie a segmentation to its target through MUTUAL INFORMATION (Zhao et al., 2019).

    Per-pixel losses treat pixels independently and ignore that neighbouring pixels are
    correlated -- the STRUCTURE a good segmentation must capture. Region mutual
    information maximises the mutual information between the predicted and the true
    label fields, which rewards getting the joint pattern right, not just each pixel.
    This soft version reads the 2x2 joint distribution of (predicted probability, true
    label) over the image and returns the negative mutual information, so minimising it
    raises the dependence between prediction and target. ``pred`` in [0,1], ``target``
    binary.
    """
    p = Tensor._wrap(pred).reshape(-1)
    y = np.asarray(target, float).ravel()
    yt = Tensor(y)
    n = len(y)
    # soft 2x2 joint distribution of (pred, target)
    p11 = (p * yt).sum() * (1.0 / n)
    p10 = (p * (1.0 - yt)).sum() * (1.0 / n)
    p01 = ((1.0 - p) * yt).sum() * (1.0 / n)
    p00 = ((1.0 - p) * (1.0 - yt)).sum() * (1.0 / n)
    pp1 = p.sum() * (1.0 / n); pp0 = 1.0 - pp1            # marginals
    py1 = float(y.mean()); py0 = 1.0 - py1
    mi = (_mi_term(p11, pp1, py1) + _mi_term(p10, pp1, py0)
          + _mi_term(p01, pp0, py1) + _mi_term(p00, pp0, py0))
    return -mi                                            # minimise -> maximise MI


__all__ = ["region_mutual_information"]
