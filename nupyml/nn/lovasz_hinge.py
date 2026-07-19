"""A differentiable surrogate for the IoU (Jaccard) loss (Berman et al., 2018)."""
import numpy as np
from ..autograd import Tensor


def _lovasz_grad(gt_sorted):
    # gradient of the Jaccard loss for the sorted ground truth (Berman et al.)
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - np.cumsum(gt_sorted)
    union = gts + np.cumsum(1 - gt_sorted)
    jaccard = 1.0 - intersection / np.maximum(union, 1e-12)
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_hinge(logits, labels):
    """A differentiable surrogate for the IoU (Jaccard) loss (Berman et al., 2018).

    Segmentation is judged by intersection-over-union, but IoU is discrete and
    non-differentiable, so people train on per-pixel cross-entropy and hope it
    correlates. Lovász-softmax optimises IoU DIRECTLY: the Jaccard set function is
    submodular, and its LOVÁSZ EXTENSION is a convex, piecewise-linear function that
    IS differentiable -- computed by sorting the pixel errors and taking a dot
    product with the gradient of the sorted Jaccard loss. Minimising it improves IoU
    where cross-entropy plateaus, especially for small foreground objects. Binary
    hinge form here (``labels`` in {0,1}, ``logits`` real).
    """
    logits = Tensor._wrap(logits).reshape(-1)
    labels = np.asarray(labels, float).ravel()
    signs = 2.0 * labels - 1.0                            # {-1, +1}
    errors = 1.0 - logits * Tensor(signs)                 # hinge margins
    order = np.argsort(-errors.data)                      # descending
    gt_sorted = labels[order]
    grad = _lovasz_grad(gt_sorted)
    errors_sorted = errors[order]
    return (errors_sorted.relu() * Tensor(grad)).sum()


__all__ = ["lovasz_hinge"]
