"""Normalization layers: keeping activations in a sane range, different ways.

WHY NORMALISE AT ALL
--------------------
As a signal passes through a deep network, its scale can drift -- growing until
gradients explode or shrinking until they vanish. Normalisation layers re-centre
and re-scale the activations at each step, which keeps gradients well-behaved and
lets you train deeper networks with larger learning rates. The methods differ in
WHAT they average over, and that single choice decides where each one works.

WHAT EACH ONE AVERAGES OVER
---------------------------
Given activations shaped (batch, channels, ...):

* ``BatchNorm`` (in ``layers.py``) -- normalise each CHANNEL across the BATCH.
  Powerful, but it couples examples together and needs a big batch, and its
  train/eval behaviour differs -- awkward for small batches and sequences.
* ``LayerNorm`` (in ``layers.py``) -- normalise each EXAMPLE across its features.
  Batch-independent, identical at train and eval -- which is why transformers use
  it and not BatchNorm.
* ``GroupNorm`` -- normalise within GROUPS of channels, per example. A middle
  ground that works at any batch size, the vision answer when the batch is small.
* ``InstanceNorm`` -- normalise each channel of each example alone. Erases
  per-image contrast, which is exactly what style transfer wants.
* ``RMSNorm`` -- LayerNorm without the mean-centering. Cheaper, and the current
  default in large language models.

The progression from BatchNorm to RMSNorm is a steady removal of dependencies:
first on the batch, then on the mean.
"""
import numpy as np

from ..autograd import Tensor
from .module import Module, Parameter


class RMSNorm(Module):
    """Root-mean-square norm: scale by the RMS, skip the mean-centering.

    LayerNorm does two things -- subtract the mean, then divide by the standard
    deviation. RMSNorm keeps only the second::

        x / sqrt(mean(x^2) + eps) * gain

    The claim, borne out in practice, is that the re-CENTERING contributes little
    -- the re-SCALING is what stabilises training. Dropping the mean removes a
    reduction and a subtraction per call, a real saving at the scale of a large
    model, with no measurable loss in quality. It is now the default norm in LLaMA
    and most recent LLMs, which is the strongest evidence that the mean term was
    doing less than everyone assumed.

    Zhang & Sennrich (2019).
    """

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gain = Parameter(np.ones(dim))     # a learned per-feature scale

    def forward(self, x):
        x = Tensor._wrap(x)
        # divide by the root-mean-square over the last axis; no mean subtracted
        ms = (x * x).mean(axis=-1, keepdims=True) if _accepts_keepdims(x) else \
            (x * x).mean(axis=-1)
        rms = (ms + self.eps).sqrt()
        if rms.ndim < x.ndim:
            rms = rms.reshape(*rms.shape, 1)
        return x / rms * self.gain


def _accepts_keepdims(x):
    try:
        x.mean(axis=-1, keepdims=True)
        return True
    except TypeError:
        return False


class GroupNorm(Module):
    """Normalise within groups of channels -- batch-independent.

    BatchNorm's statistics come from the whole batch, so with a batch of 2 (common
    in high-resolution vision) they are too noisy to be useful and it degrades
    badly. GroupNorm computes its mean and variance WITHIN each example, over a
    GROUP of channels, so it does not depend on the batch at all and works
    identically at any batch size and at eval time.

    ``num_groups`` interpolates between the other norms: one group is InstanceNorm
    of the whole feature vector, ``num_channels`` groups is per-channel. The usual
    default of 32 groups is the vision standard when BatchNorm's batch dependence
    is a problem.

    Wu & He (2018).
    """

    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by "
                f"num_groups ({num_groups})")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.gamma = Parameter(np.ones(num_channels))
        self.beta = Parameter(np.zeros(num_channels))

    def forward(self, x):
        # expects (batch, channels, ...) -- reshape so each group is one axis to
        # reduce over, per example
        x = Tensor._wrap(x)
        N, C = x.shape[0], x.shape[1]
        spatial = x.shape[2:]
        g = self.num_groups
        xr = x.reshape(N, g, C // g, *spatial)
        # reduce over everything but the batch and group axes
        axes = tuple(range(2, xr.ndim))
        mean = xr.data.mean(axis=axes, keepdims=True)
        var = xr.data.var(axis=axes, keepdims=True)
        normed = (xr - Tensor(mean)) / (Tensor(var) + self.eps).sqrt()
        out = normed.reshape(N, C, *spatial)
        # per-channel affine, broadcast over the spatial dims
        shape = [1, C] + [1] * len(spatial)
        return out * self.gamma.reshape(*shape) + self.beta.reshape(*shape)


class InstanceNorm(Module):
    """Normalise each channel of each example independently -- for style.

    GroupNorm with one channel per group: it standardises every feature map on
    its own, per image. This ERASES the per-image mean and contrast of each
    channel -- which for most tasks throws away signal, but for STYLE TRANSFER is
    the whole point. Style is largely carried by those channel statistics, so
    normalising them out and re-injecting a target style's statistics is how you
    restyle an image. The right tool depends entirely on whether that information
    is signal or nuisance.

    Ulyanov, Vedaldi & Lempitsky (2016).
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = Parameter(np.ones(num_features))
        self.beta = Parameter(np.zeros(num_features))

    def forward(self, x):
        x = Tensor._wrap(x)
        spatial = tuple(range(2, x.ndim))       # normalise over spatial dims only
        mean = x.data.mean(axis=spatial, keepdims=True)
        var = x.data.var(axis=spatial, keepdims=True)
        normed = (x - Tensor(mean)) / (Tensor(var) + self.eps).sqrt()
        C = x.shape[1]
        shape = [1, C] + [1] * len(spatial)
        return normed * self.gamma.reshape(*shape) + self.beta.reshape(*shape)


__all__ = ["RMSNorm", "GroupNorm", "InstanceNorm"]
