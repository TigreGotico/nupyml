"""Neural network layers."""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from ..utils import check_random_state
from .module import Module, Parameter
from . import init


class Linear(Module):
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
    def __init__(self, num_embeddings, embedding_dim, rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self.weight = Parameter(rng.normal(0, 1.0, size=(num_embeddings, embedding_dim)))

    def forward(self, indices):
        return F.embedding(indices, self.weight)


class Conv2d(Module):
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
