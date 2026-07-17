"""Regularizers that fight overfitting by DISTORTING training, not the loss.

Weight decay and dropout shrink or drop parameters. The methods here take a
different route: they corrupt the DATA or the ARCHITECTURE during training so the
model cannot rely on any single input pattern or any single layer. Each is a
different answer to "make the model robust by never letting it get comfortable".
"""
import numpy as np

from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module


def mixup(X, y, alpha=0.2, n_classes=None, rng=None):
    """Train on BLENDS of pairs of examples, with blended labels.

    THE IDEA
    --------
    Take two random training examples and mix them::

        x_mixed = lam * x_i + (1 - lam) * x_j
        y_mixed = lam * y_i + (1 - lam) * y_j     (one-hot labels)

    with ``lam`` drawn from a Beta distribution. The model must predict the SAME
    blend for the mixed input -- so it is forced to behave LINEARLY between
    training points, rather than carving sharp, overconfident regions around each
    one.

    WHY IT HELPS
    ------------
    Networks left alone become overconfident far from the data and brittle to
    small perturbations. Mixup fills the space BETWEEN examples with sensible
    interpolated targets, which smooths the decision boundary, calibrates the
    confidence, and improves robustness -- all from a two-line data transform, no
    change to the model or loss. ``alpha`` controls the blend strength; small
    values mix gently.

    Zhang et al. (2017).
    """
    rng = check_random_state(rng)
    X = np.asarray(X, float)
    y = np.asarray(y)
    n = len(X)
    lam = rng.beta(alpha, alpha)                 # the mixing coefficient
    perm = rng.permutation(n)                    # pair each example with another

    X_mixed = lam * X + (1 - lam) * X[perm]
    # labels must be one-hot to blend -- "70% cat, 30% dog" has no meaning as an int
    if y.ndim == 1:
        k = n_classes or int(y.max() + 1)
        y_oh = np.eye(k)[y]
    else:
        y_oh = y
    y_mixed = lam * y_oh + (1 - lam) * y_oh[perm]
    return X_mixed, y_mixed, lam


def cutmix(X, y, alpha=1.0, n_classes=None, rng=None):
    """Paste a PATCH of one image onto another; mix labels by patch area.

    THE VARIATION ON MIXUP
    ----------------------
    Mixup blends two whole images into a translucent average, which looks
    unnatural. CutMix instead CUTS a rectangular region from one image and pastes
    it over another, and mixes the labels in proportion to the pasted AREA. The
    result is a locally-coherent image (real pixels everywhere, just from two
    sources), which suits convolutional models better than mixup's ghosted blend.

    It also acts like a smarter cutout: the occluded region is replaced by REAL
    content from another class rather than zeros, so no training signal is wasted.
    Expects image tensors shaped (batch, channels, height, width).

    Yun et al. (2019).
    """
    rng = check_random_state(rng)
    X = np.asarray(X, float).copy()
    y = np.asarray(y)
    n, _, H, W = X.shape
    lam = rng.beta(alpha, alpha)
    perm = rng.permutation(n)

    # a patch whose area is (1 - lam) of the image, placed at a random centre
    cut_ratio = np.sqrt(1 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cy, cx = rng.randint(H), rng.randint(W)
    y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, H)
    x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, W)
    X[:, :, y1:y2, x1:x2] = X[perm, :, y1:y2, x1:x2]

    # the true mixing is the ACTUAL pasted-area fraction, not the sampled lam
    lam_adj = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    if y.ndim == 1:
        k = n_classes or int(y.max() + 1)
        y_oh = np.eye(k)[y]
    else:
        y_oh = y
    y_mixed = lam_adj * y_oh + (1 - lam_adj) * y_oh[perm]
    return X, y_mixed, lam_adj


class StochasticDepth(Module):
    """Randomly SKIP an entire residual block during training.

    THE IDEA
    --------
    Dropout drops neurons; stochastic depth (a.k.a. DropPath) drops whole LAYERS.
    On each training step a residual block is bypassed -- replaced by its identity
    shortcut -- with some probability, so the network trains as a random ensemble
    of shallower networks that share weights.

    WHY IT WORKS FOR VERY DEEP NETS
    -------------------------------
    Two payoffs. Every block must be useful ON ITS OWN, since it cannot count on
    any specific later block being present -- the same anti-co-adaptation logic as
    dropout, at layer granularity. And the expected depth during training is
    shorter, so gradients flow more easily, which is what let networks reach a
    thousand-plus layers. At test time all blocks are kept but scaled by their
    survival probability, exactly as with dropout.

    Huang et al. (2016).
    """

    def __init__(self, block, survival_prob=0.8, rng=None):
        super().__init__()
        self.block = block
        self.survival_prob = survival_prob
        self._rng = check_random_state(rng)

    def forward(self, x):
        x = Tensor._wrap(x)
        if not self.training:
            # at eval, keep the block but scale by survival prob, so the expected
            # contribution matches training
            return x + self.block(x) * self.survival_prob
        if self._rng.uniform() < self.survival_prob:
            return x + self.block(x)            # block survives this step
        return x                                # block dropped: identity only


def drop_path(x, drop_prob, training, rng=None):
    """Per-EXAMPLE stochastic depth: drop the block for some rows of the batch.

    StochasticDepth drops a block for the whole batch; drop_path decides per
    example, which gives finer-grained regularisation and is the form used inside
    modern vision transformers. Surviving rows are scaled up by ``1/keep`` so the
    expected output is unchanged -- the same inverted-dropout bookkeeping.
    """
    if not training or drop_prob == 0:
        return x
    rng = check_random_state(rng)
    x = Tensor._wrap(x)
    keep = 1 - drop_prob
    # one keep/drop mask per example, broadcast over the feature dims
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = (rng.uniform(size=shape) < keep).astype(np.float64) / keep
    return x * Tensor(mask)


__all__ = ["mixup", "cutmix", "StochasticDepth", "drop_path"]
