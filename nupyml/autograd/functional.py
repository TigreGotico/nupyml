"""Higher-level differentiable functions: softmax, conv2d, pooling, embedding."""
import numpy as np

from .tensor import Tensor


def softmax(x, axis=-1):
    x = Tensor._wrap(x)
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    out_data = e / e.sum(axis=axis, keepdims=True)
    def backward(g):
        if x.requires_grad:
            # dL/dx = s * (g - sum(g*s))
            dot = (g * out_data).sum(axis=axis, keepdims=True)
            x._accumulate(out_data * (g - dot))
    return Tensor._make(out_data, (x,), backward, "softmax")


def log_softmax(x, axis=-1):
    x = Tensor._wrap(x)
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    out_data = shifted - logsumexp
    s = np.exp(out_data)
    def backward(g):
        if x.requires_grad:
            x._accumulate(g - s * g.sum(axis=axis, keepdims=True))
    return Tensor._make(out_data, (x,), backward, "log_softmax")


def leaky_relu(x, negative_slope=0.01):
    x = Tensor._wrap(x)
    mask = x.data > 0
    scale = np.where(mask, 1.0, negative_slope)
    def backward(g):
        if x.requires_grad:
            x._accumulate(g * scale)
    return Tensor._make(x.data * scale, (x,), backward, "leaky_relu")


def gelu(x):
    """Gaussian Error Linear Unit (tanh approximation)."""
    x = Tensor._wrap(x)
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x.data + 0.044715 * x.data ** 3)
    t = np.tanh(inner)
    out_data = 0.5 * x.data * (1 + t)
    def backward(g):
        if x.requires_grad:
            dinner = c * (1 + 3 * 0.044715 * x.data ** 2)
            dt = (1 - t ** 2) * dinner
            x._accumulate(g * (0.5 * (1 + t) + 0.5 * x.data * dt))
    return Tensor._make(out_data, (x,), backward, "gelu")


# ---------------------------------------------------------------------------
# im2col-based convolution
# ---------------------------------------------------------------------------

def _im2col_indices(x_shape, kh, kw, stride, padding):
    n, c, h, w = x_shape
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1
    i0 = np.repeat(np.arange(kh), kw)
    i0 = np.tile(i0, c)
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    j0 = np.tile(np.arange(kw), kh * c)
    j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)   # (c*kh*kw, out_h*out_w)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c), kh * kw).reshape(-1, 1)
    return k, i, j, out_h, out_w


def _im2col(x, kh, kw, stride, padding):
    p = padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p > 0 else x
    k, i, j, out_h, out_w = _im2col_indices(x.shape, kh, kw, stride, padding)
    cols = x_pad[:, k, i, j]                     # (n, c*kh*kw, out_h*out_w)
    return cols, out_h, out_w


def _col2im(cols, x_shape, kh, kw, stride, padding):
    n, c, h, w = x_shape
    p = padding
    x_pad = np.zeros((n, c, h + 2 * p, w + 2 * p))
    k, i, j, _, _ = _im2col_indices(x_shape, kh, kw, stride, padding)
    np.add.at(x_pad, (slice(None), k, i, j), cols)
    return x_pad[:, :, p:p + h, p:p + w] if p > 0 else x_pad


def conv2d(x, weight, bias=None, stride=1, padding=0):
    """2D convolution. x: (N,C,H,W), weight: (F,C,KH,KW), bias: (F,)."""
    x, weight = Tensor._wrap(x), Tensor._wrap(weight)
    n, c, h, w = x.data.shape
    f, _, kh, kw = weight.data.shape
    cols, out_h, out_w = _im2col(x.data, kh, kw, stride, padding)  # (n, c*kh*kw, L)
    w_row = weight.data.reshape(f, -1)                             # (f, c*kh*kw)
    out_data = np.einsum("fk,nkl->nfl", w_row, cols).reshape(n, f, out_h, out_w)
    parents = [x, weight]
    if bias is not None:
        bias = Tensor._wrap(bias)
        out_data = out_data + bias.data.reshape(1, f, 1, 1)
        parents.append(bias)

    def backward(g):
        g2 = g.reshape(n, f, -1)                                   # (n, f, L)
        if weight.requires_grad:
            gw = np.einsum("nfl,nkl->fk", g2, cols).reshape(weight.data.shape)
            weight._accumulate(gw)
        if bias is not None and bias.requires_grad:
            bias._accumulate(g2.sum(axis=(0, 2)))
        if x.requires_grad:
            gcols = np.einsum("fk,nfl->nkl", w_row, g2)
            gx = _col2im(gcols, x.data.shape, kh, kw, stride, padding)
            x._accumulate(gx)

    return Tensor._make(out_data, parents, backward, "conv2d")


def max_pool2d(x, kernel_size=2, stride=None):
    x = Tensor._wrap(x)
    k = kernel_size
    s = stride or k
    n, c, h, w = x.data.shape
    out_h = (h - k) // s + 1
    out_w = (w - k) // s + 1
    # strided view of pooling windows
    strides = x.data.strides
    windows = np.lib.stride_tricks.as_strided(
        x.data,
        shape=(n, c, out_h, out_w, k, k),
        strides=(strides[0], strides[1], strides[2] * s, strides[3] * s,
                 strides[2], strides[3]),
    )
    flat = windows.reshape(n, c, out_h, out_w, k * k)
    argmax = flat.argmax(axis=-1)
    out_data = np.take_along_axis(flat, argmax[..., None], axis=-1)[..., 0]

    def backward(g):
        if not x.requires_grad:
            return
        gx = np.zeros_like(x.data)
        # scatter grads back to argmax positions
        ii, jj = np.unravel_index(argmax, (k, k))
        ni, ci, oi, oj = np.indices((n, c, out_h, out_w))
        np.add.at(gx, (ni, ci, oi * s + ii, oj * s + jj), g)
        x._accumulate(gx)

    return Tensor._make(out_data, (x,), backward, "max_pool2d")


def avg_pool2d(x, kernel_size=2, stride=None):
    x = Tensor._wrap(x)
    k = kernel_size
    s = stride or k
    n, c, h, w = x.data.shape
    out_h = (h - k) // s + 1
    out_w = (w - k) // s + 1
    strides = x.data.strides
    windows = np.lib.stride_tricks.as_strided(
        x.data,
        shape=(n, c, out_h, out_w, k, k),
        strides=(strides[0], strides[1], strides[2] * s, strides[3] * s,
                 strides[2], strides[3]),
    )
    out_data = windows.mean(axis=(-1, -2))

    def backward(g):
        if not x.requires_grad:
            return
        gx = np.zeros_like(x.data)
        gshare = g / (k * k)
        for di in range(k):
            for dj in range(k):
                gx_view = gx[:, :, di:di + out_h * s:s, dj:dj + out_w * s:s]
                gx_view += gshare
        x._accumulate(gx)

    return Tensor._make(out_data, (x,), backward, "avg_pool2d")


def embedding(indices, weight):
    """Lookup rows of ``weight`` (V, D) by integer ``indices``."""
    weight = Tensor._wrap(weight)
    idx = np.asarray(indices, dtype=np.int64)
    def backward(g):
        if weight.requires_grad:
            gw = np.zeros_like(weight.data)
            np.add.at(gw, idx.ravel(), g.reshape(-1, weight.data.shape[1]))
            weight._accumulate(gw)
    return Tensor._make(weight.data[idx], (weight,), backward, "embedding")


def dropout(x, p=0.5, training=True, rng=None):
    x = Tensor._wrap(x)
    if not training or p == 0.0:
        return x
    rng = rng or np.random
    mask = (rng.uniform(size=x.data.shape) >= p) / (1.0 - p)
    def backward(g):
        if x.requires_grad:
            x._accumulate(g * mask)
    return Tensor._make(x.data * mask, (x,), backward, "dropout")


__all__ = ["softmax", "log_softmax", "leaky_relu", "gelu", "conv2d",
           "max_pool2d", "avg_pool2d", "embedding", "dropout"]
