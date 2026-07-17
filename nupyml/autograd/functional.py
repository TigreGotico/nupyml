"""Differentiable functions built on the Tensor primitives.

These could all be written in terms of ``+``, ``*`` and friends and let the
tape figure out the gradient. They are hand-written instead for two reasons
that recur throughout numerical computing:

*Numerical stability.* ``softmax`` written naively overflows on inputs a
human would call unremarkable. Composing stable pieces does not automatically
give a stable whole.

*Efficiency.* A convolution expressed as a Python loop over pixels would
record thousands of tiny graph nodes. Reshaping it into one matmul records
one, and hands the arithmetic to BLAS.
"""
import numpy as np

from .tensor import Tensor


def softmax(x, axis=-1):
    """Turn arbitrary scores into a probability distribution.

    WHY THE SHIFT
    -------------
    The definition is ``exp(x_i) / sum_j exp(x_j)``. Computed literally, a score
    of 1000 overflows to ``inf`` and the result is ``nan`` -- and 1000 is not an
    exotic logit.

    Subtracting the row maximum first fixes it exactly, with no approximation.
    Multiply numerator and denominator by ``exp(-max)``::

        exp(x_i - max) / sum_j exp(x_j - max)

    Identical value, but now the largest exponent is ``exp(0) == 1``, so nothing
    can overflow, and underflow to zero only happens for terms that were
    negligible anyway. This shift-invariance is a property of softmax itself:
    adding a constant to every score changes nothing.

    THE GRADIENT
    ------------
    The true Jacobian is dense -- every output depends on every input, since
    they share a denominator::

        ds_i/dx_j = s_i * (delta_ij - s_j)

    Materialising that per row would cost O(k^2). But the VJP collapses to::

        dL/dx = s * (g - sum(g * s))

    which is O(k). The subtracted term is the "competition" between classes:
    pushing one probability up necessarily pushes the others down.
    """
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
    """``log(softmax(x))``, computed as one stable operation.

    Never as ``softmax(x).log()``. Softmax outputs underflow to exactly 0.0 for
    confidently-wrong classes, and ``log(0)`` is ``-inf``, which poisons the
    loss and every gradient downstream. Keeping it in log space avoids ever
    forming the small number: the log of a sum of exponentials is evaluated
    directly, so a tiny probability becomes a large negative number -- which is
    perfectly representable -- instead of zero.

    The gradient is also strikingly simpler than softmax's::

        dL/dx = g - softmax(x) * sum(g)

    This is why cross-entropy is implemented as log_softmax followed by a
    lookup: the notorious ``p - y`` gradient of the pair falls out of that
    expression, and neither stage ever handles a raw probability.
    """
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
    """Gaussian Error Linear Unit, the transformer's activation.

    Where ReLU makes a hard decision -- keep it or kill it -- GELU weights the
    input by the probability that a standard normal falls below it:
    ``x * Phi(x)``. Small inputs are damped rather than deleted, giving a
    smooth curve with a gradient everywhere and no dead region.

    ``Phi`` has no elementary closed form, so the standard tanh approximation
    is used; it is accurate to a few decimals and much cheaper than ``erf``.
    """
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
    """Precompute where each patch element lives in the padded image.

    Builds three broadcast index arrays -- channel, row, column -- such that
    ``x_padded[:, k, i, j]`` gathers every patch at once. Fancy indexing does
    the copying in C; the alternative is a Python loop per output pixel.
    """
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
    """Scatter patch gradients back onto the image (the adjoint of im2col).

    ``np.add.at`` rather than plain indexed assignment is essential here:
    patches overlap, so the same pixel appears at several places in the index
    arrays. Assignment would keep only the last write; ``add.at`` accumulates
    every contribution, which is what the chain rule requires.
    """
    n, c, h, w = x_shape
    p = padding
    x_pad = np.zeros((n, c, h + 2 * p, w + 2 * p))
    k, i, j, _, _ = _im2col_indices(x_shape, kh, kw, stride, padding)
    np.add.at(x_pad, (slice(None), k, i, j), cols)
    return x_pad[:, :, p:p + h, p:p + w] if p > 0 else x_pad


def conv2d(x, weight, bias=None, stride=1, padding=0):
    """2D convolution, as a single matrix multiply.

    THE IDEA: CONVOLUTION IS ALREADY A MATMUL IN DISGUISE
    -----------------------------------------------------
    At each output position, a conv computes a dot product between a filter and
    the patch of input under it. Doing this with four nested loops (positions x
    filters) is correct and unbearably slow in Python.

    ``im2col`` makes the matmul explicit. Copy every patch out into a column of
    a big matrix::

        cols:  (N, C*KH*KW, L)      L = number of output positions
        w_row: (F, C*KH*KW)         each filter flattened to a row

    Then the entire convolution -- every filter against every position -- is one
    product, dispatched to BLAS.

    THE COST
    --------
    ``cols`` duplicates data: with a 3x3 kernel, each input pixel is copied into
    up to 9 patches, so memory grows ~9x. This is a deliberate trade of memory
    for speed, and it is what essentially every framework did before hand-tuned
    kernels. Being able to see the trade is the point of writing it this way.

    THE GRADIENT
    ------------
    Because the forward pass is a matmul, the gradients are the matmul VJPs::

        dL/dw    = dL/dout @ cols'
        dL/dcols = w' @ dL/dout

    The only new piece is ``col2im``: scattering ``dL/dcols`` back to pixels.
    Each input pixel appeared in several patches, so it accumulates a
    contribution from each -- the same "reuse means sum" rule as broadcasting.

    Parameters
    ----------
    x : Tensor (N, C, H, W)
    weight : Tensor (F, C, KH, KW)
    bias : Tensor (F,), optional
    """
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
    """Downsample by keeping the strongest activation in each window.

    The forward pass uses a strided view -- ``as_strided`` re-describes the
    existing buffer with new shape and strides, materialising the sliding
    windows with zero copying. It is the sharpest tool in numpy: the strides
    are taken on trust, and wrong ones read out of bounds silently. Here they
    are derived directly from the array's own strides, so the windows are
    guaranteed to lie inside it.

    Backward is a scatter. Pooling selects, so the gradient goes entirely to
    the argmax of each window and zero everywhere else -- which is why the
    argmax is saved on the forward pass rather than recomputed.
    """
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
    """Look up rows of a table by integer index.

    An embedding layer is exactly a one-hot vector times a weight matrix -- but
    that product is almost entirely multiplication by zero, so it is done as a
    row lookup instead.

    Backward is where the equivalence shows: the gradient is scattered back to
    the rows that were used, and ``np.add.at`` is required because a token
    appearing multiple times in a batch must accumulate a contribution per
    occurrence. Rows never looked up receive nothing and stay unchanged -- which
    is why embedding gradients are naturally sparse.
    """
    weight = Tensor._wrap(weight)
    idx = np.asarray(indices, dtype=np.int64)
    def backward(g):
        if weight.requires_grad:
            gw = np.zeros_like(weight.data)
            np.add.at(gw, idx.ravel(), g.reshape(-1, weight.data.shape[1]))
            weight._accumulate(gw)
    return Tensor._make(weight.data[idx], (weight,), backward, "embedding")


def dropout(x, p=0.5, training=True, rng=None):
    """Randomly zero activations during training.

    Dropout stops units from co-adapting: a unit cannot rely on a specific
    partner being present, so the network is pushed toward redundant,
    independently-useful features. It is loosely an ensemble over the
    exponentially many sub-networks obtainable by deleting units.

    THE 1/(1-p) SCALING
    -------------------
    Kept activations are divided by ``1 - p`` ("inverted dropout"). Without it,
    a layer's expected input magnitude would drop by a factor of ``1 - p``
    during training but not at test time, and the shift would compound through
    depth. Scaling up during training keeps the expectation matched, so test
    time needs no adjustment at all -- ``eval()`` simply returns the input
    untouched.

    The mask is a plain constant as far as the tape is concerned, so backward
    applies the same mask: gradients flow only through units that participated.
    """
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
