"""Layers: the building blocks a network is assembled from.

Almost every layer here exists to solve a problem that showed up when people
tried to make networks deeper.

WHY NON-LINEARITIES ARE NOT OPTIONAL
------------------------------------
Stack two Linear layers and you get ``(xW1 + b1)W2 + b2``, which multiplies out
to ``xW' + b'`` -- one Linear layer. Without something non-linear between them,
depth buys literally nothing. ``ReLU`` and friends are what make a deep network
more expressive than a shallow one, not merely larger.

WHY NORMALIZATION LAYERS EXIST
------------------------------
Each layer's input distribution shifts as the layers below it learn, so every
layer is chasing a moving target. Normalising the activations pins the scale, so
gradients neither explode nor vanish through depth, and much higher learning
rates become safe. ``BatchNorm`` normalises across the batch, ``LayerNorm``
across the features -- and the difference matters; see their docstrings.

WHY CONVOLUTIONS EXIST
----------------------
A Linear layer on a 224x224 image would need ~50k weights PER unit and would
have to learn what an edge is separately at every position. A ``Conv2d`` slides
one small filter everywhere, which encodes two priors about images directly
into the architecture: locality (nearby pixels are related) and translation
equivariance (a cat is a cat wherever it appears). Fewer parameters and a
better inductive bias.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from ..utils import check_random_state
from .module import Module, Parameter
from . import init


class Linear(Module):
    """A fully connected layer: ``y = x @ W + b``.

    Every input influences every output. This is the general case, and the
    reason it is not used everywhere is cost: parameters grow as
    ``in x out``, and it assumes no structure in the input at all -- shuffling
    the input features would make no difference to what it can learn, which for
    an image is throwing away everything you know.
    """

    def __init__(self, in_features, out_features, bias=True, rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self.weight = Parameter(init.kaiming_uniform((in_features, out_features), rng))
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope)


class GELU(Module):
    def forward(self, x):
        return F.gelu(x)


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return F.softmax(x, axis=self.axis)


class Flatten(Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Dropout(Module):
    def __init__(self, p=0.5, rng=None):
        super().__init__()
        self.p = p
        self.rng = check_random_state(rng)

    def forward(self, x):
        return F.dropout(x, p=self.p, training=self.training, rng=self.rng)


class Embedding(Module):
    """A learned vector per discrete token.

    One-hot encoding a 50k vocabulary gives 50k sparse dimensions in which every
    pair of words is equidistant -- "cat" is exactly as far from "dog" as from
    "thermodynamics". An embedding maps each token to a dense learned vector
    instead, so similar tokens can end up nearby, and the model learns that
    geometry from the task.

    Mechanically it is a one-hot vector times a weight matrix, implemented as a
    row lookup because the product is almost all zeros.
    """

    def __init__(self, num_embeddings, embedding_dim, rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self.weight = Parameter(rng.normal(0, 1.0, size=(num_embeddings, embedding_dim)))

    def forward(self, indices):
        return F.embedding(indices, self.weight)


class Conv2d(Module):
    """Slide learned filters over an image.

    Each filter is small (typically 3x3) and is applied at every position, so
    the same weights are reused across the whole image -- which is exactly what
    "translation equivariance" means, and what makes conv layers so
    parameter-efficient compared to Linear.

    Each of the ``out_channels`` filters spans ALL input channels: a filter is
    ``(in_channels, kh, kw)`` and produces one output channel. So channels are
    fully connected while space is local -- the network is told that position
    matters and channel identity does not.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, rng=None):
        super().__init__()
        rng = check_random_state(rng)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        bound = np.sqrt(6.0 / fan_in)
        self.weight = Parameter(rng.uniform(
            -bound, bound, size=(out_channels, in_channels, *kernel_size)))
        self.bias = Parameter(np.zeros(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=self.stride,
                        padding=self.padding)


class MaxPool2d(Module):
    """Downsample by keeping the largest activation per window.

    Two things at once: the spatial size shrinks (cheaper, and later layers see
    a wider receptive field), and small translations stop mattering -- shifting a
    feature by one pixel usually leaves the window maximum unchanged. That is
    local translation INVARIANCE, on top of convolution's equivariance.

    Max rather than average because a feature map answers "is this pattern
    present?", and the strongest response is the evidence. Averaging would
    dilute it with the surrounding silence.
    """

    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride)


class AvgPool2d(Module):
    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride)


class BatchNorm1d(Module):
    """Normalise each feature across the batch, then re-scale it learnably.

    Subtract the batch mean, divide by the batch standard deviation, then apply
    a learned ``weight`` and ``bias``. The learned pair matters: forcing every
    activation to mean 0 / variance 1 would be a straitjacket -- a sigmoid would
    be pinned to its linear region -- so the network is given the parameters to
    undo the normalisation if that is what it wants. It gets a stable
    STARTING scale, not a mandatory one.

    TRAIN AND EVAL DIFFER, AND MUST
    -------------------------------
    Training normalises by the statistics of the CURRENT batch. At test time
    there may be no batch -- predictions must not depend on which other samples
    happen to be alongside -- so a running average of the training statistics is
    used instead. This is why ``train()`` and ``eval()`` exist, and why
    forgetting ``eval()`` produces mysteriously wrong predictions.

    The batch dependence is also BatchNorm's weakness: with small batches the
    statistics are noisy, and for sequences of varying length they are
    ill-defined. That is what ``LayerNorm`` is for.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(np.ones(num_features))
        self.bias = Parameter(np.zeros(num_features))
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x):
        if self.training:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = ((1 - self.momentum) * self.running_mean
                                 + self.momentum * mu.data)
            self.running_var = ((1 - self.momentum) * self.running_var
                                + self.momentum * var.data)
        else:
            mu = Tensor(self.running_mean)
            var = Tensor(self.running_var)
        xhat = (x - mu) / ((var + self.eps) ** 0.5)
        return xhat * self.weight + self.bias


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(np.ones(num_features))
        self.bias = Parameter(np.zeros(num_features))
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x):
        axes = (0, 2, 3)
        if self.training:
            mu = x.mean(axis=axes, keepdims=True)
            var = x.var(axis=axes, keepdims=True)
            self.running_mean = ((1 - self.momentum) * self.running_mean
                                 + self.momentum * mu.data.ravel())
            self.running_var = ((1 - self.momentum) * self.running_var
                                + self.momentum * var.data.ravel())
        else:
            mu = Tensor(self.running_mean.reshape(1, -1, 1, 1))
            var = Tensor(self.running_var.reshape(1, -1, 1, 1))
        xhat = (x - mu) / ((var + self.eps) ** 0.5)
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        return xhat * w + b


class LayerNorm(Module):
    """Normalise each sample across its own features.

    The axis is the whole difference from BatchNorm, and it is a big one: each
    sample is normalised using ONLY itself. So there is no batch dependence, no
    train/eval divergence, no running statistics, and it works identically for a
    batch of 1 or 1000, and for sequences of any length.

    That is why transformers use LayerNorm and convnets use BatchNorm: a
    transformer's inputs are variable-length sequences, where per-batch
    statistics over padded positions would be meaningless.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = Parameter(np.ones(normalized_shape))
        self.bias = Parameter(np.zeros(normalized_shape))
        self._axes = tuple(range(-len(normalized_shape), 0))

    def forward(self, x):
        mu = x.mean(axis=self._axes, keepdims=True)
        var = x.var(axis=self._axes, keepdims=True)
        xhat = (x - mu) / ((var + self.eps) ** 0.5)
        return xhat * self.weight + self.bias


__all__ = ["Linear", "ReLU", "LeakyReLU", "GELU", "Tanh", "Sigmoid", "Softmax",
           "Flatten", "Dropout", "Embedding", "Conv2d", "MaxPool2d", "AvgPool2d",
           "BatchNorm1d", "BatchNorm2d", "LayerNorm"]
