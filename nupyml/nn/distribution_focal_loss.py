"""Learn a DISTRIBUTION over box offsets, not a point (Li et al., 2020)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F


def distribution_focal_loss(logits, target, bin_edges):
    """Learn a DISTRIBUTION over box offsets, not a point (Li et al., 2020).

    A detector usually regresses a box edge to a single number, which cannot express
    that a blurry or occluded boundary is UNCERTAIN. Distribution focal loss discretises
    the offset into bins and predicts a probability over them, so the model can be
    sharp on clear edges and spread on ambiguous ones (and the expectation is the
    prediction). For a continuous target between two adjacent bins, it is the cross-
    entropy that pushes probability onto exactly those two bins in proportion to the
    target's position between them. ``logits`` is ``(batch, n_bins)``.
    """
    logits = Tensor._wrap(logits)
    edges = np.asarray(bin_edges, float)
    target = np.asarray(target, float).ravel()
    logp = F.log_softmax(logits, axis=1)
    n = len(target)
    left = np.clip(np.searchsorted(edges, target, "right") - 1, 0, len(edges) - 2)
    right = left + 1
    wl = (edges[right] - target) / (edges[right] - edges[left])
    wr = 1.0 - wl
    onehot_l = np.zeros((n, logits.shape[1])); onehot_l[np.arange(n), left] = 1
    onehot_r = np.zeros((n, logits.shape[1])); onehot_r[np.arange(n), right] = 1
    target_dist = Tensor(onehot_l) * Tensor(wl.reshape(-1, 1)) \
        + Tensor(onehot_r) * Tensor(wr.reshape(-1, 1))
    return -(target_dist * logp).sum(axis=1).mean()


__all__ = ["distribution_focal_loss"]
