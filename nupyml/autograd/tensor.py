"""The Tensor class: an ndarray with a gradient tape."""
import contextlib

import numpy as np

_grad_enabled = True


@contextlib.contextmanager
def no_grad():
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev


def is_grad_enabled():
    return _grad_enabled


def _unbroadcast(grad, shape):
    """Sum ``grad`` over axes that were broadcast to reach ``grad.shape``."""
    if grad.shape == shape:
        return grad
    # sum over leading extra dims
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    # sum over dims that were 1 in the original shape
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A numpy array plus autodiff bookkeeping."""

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")
    __array_priority__ = 100  # numpy defers to our __radd__ etc.

    def __init__(self, data, requires_grad=False, _prev=(), _op=""):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None
        self.requires_grad = requires_grad and _grad_enabled
        self._backward = None
        self._prev = _prev if self.requires_grad or any(
            isinstance(p, Tensor) and p.requires_grad for p in _prev) else ()
        self._op = _op

    # ------------------------------------------------------------------
    # graph construction helper
    # ------------------------------------------------------------------
    @staticmethod
    def _make(data, parents, backward, op=""):
        req = _grad_enabled and any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=req)
        if req:
            out._prev = tuple(parents)
            out._backward = backward
            out._op = op
        return out

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return self.transpose()

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad})"

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def detach(self):
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self):
        self.grad = None

    # ------------------------------------------------------------------
    # backward
    # ------------------------------------------------------------------
    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Called backward on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensors")
            grad = np.ones_like(self.data)
        else:
            grad = np.asarray(grad, dtype=np.float64)

        # topological order
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if p.requires_grad and id(p) not in visited:
                    stack.append((p, False))

        self.grad = grad if self.grad is None else self.grad + grad
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)

    def _accumulate(self, grad):
        grad = _unbroadcast(np.asarray(grad, dtype=np.float64), self.data.shape)
        self.grad = grad if self.grad is None else self.grad + grad

    # ------------------------------------------------------------------
    # elementwise arithmetic
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap(other):
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other):
        other = self._wrap(other)
        def backward(g):
            if self.requires_grad:
                self._accumulate(g)
            if other.requires_grad:
                other._accumulate(g)
        return Tensor._make(self.data + other.data, (self, other), backward, "+")

    __radd__ = __add__

    def __neg__(self):
        def backward(g):
            if self.requires_grad:
                self._accumulate(-g)
        return Tensor._make(-self.data, (self,), backward, "neg")

    def __sub__(self, other):
        other = self._wrap(other)
        def backward(g):
            if self.requires_grad:
                self._accumulate(g)
            if other.requires_grad:
                other._accumulate(-g)
        return Tensor._make(self.data - other.data, (self, other), backward, "-")

    def __rsub__(self, other):
        return self._wrap(other) - self

    def __mul__(self, other):
        other = self._wrap(other)
        def backward(g):
            if self.requires_grad:
                self._accumulate(g * other.data)
            if other.requires_grad:
                other._accumulate(g * self.data)
        return Tensor._make(self.data * other.data, (self, other), backward, "*")

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._wrap(other)
        def backward(g):
            if self.requires_grad:
                self._accumulate(g / other.data)
            if other.requires_grad:
                other._accumulate(-g * self.data / (other.data ** 2))
        return Tensor._make(self.data / other.data, (self, other), backward, "/")

    def __rtruediv__(self, other):
        return self._wrap(other) / self

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only scalar powers supported"
        def backward(g):
            if self.requires_grad:
                self._accumulate(g * exponent * self.data ** (exponent - 1))
        return Tensor._make(self.data ** exponent, (self,), backward, "pow")

    def __matmul__(self, other):
        other = self._wrap(other)
        a, b = self.data, other.data
        def backward(g):
            if self.requires_grad:
                if b.ndim == 1:
                    ga = np.outer(g, b) if a.ndim == 2 else g[..., None] * b
                elif a.ndim == 1:
                    ga = g @ np.swapaxes(b, -1, -2)
                else:
                    ga = g @ np.swapaxes(b, -1, -2)
                self._accumulate(ga.reshape(a.shape) if ga.shape != a.shape and ga.size == a.size else ga)
            if other.requires_grad:
                if a.ndim == 1:
                    gb = np.outer(a, g) if b.ndim == 2 else a[..., None] * g
                elif b.ndim == 1:
                    gb = np.swapaxes(a, -1, -2) @ g
                else:
                    gb = np.swapaxes(a, -1, -2) @ g
                other._accumulate(gb.reshape(b.shape) if gb.shape != b.shape and gb.size == b.size else gb)
        return Tensor._make(a @ b, (self, other), backward, "@")

    def __rmatmul__(self, other):
        return self._wrap(other) @ self

    # comparisons produce plain arrays (no grad)
    def __gt__(self, other):
        return self.data > (other.data if isinstance(other, Tensor) else other)

    def __lt__(self, other):
        return self.data < (other.data if isinstance(other, Tensor) else other)

    def __ge__(self, other):
        return self.data >= (other.data if isinstance(other, Tensor) else other)

    def __le__(self, other):
        return self.data <= (other.data if isinstance(other, Tensor) else other)

    # ------------------------------------------------------------------
    # unary math
    # ------------------------------------------------------------------
    def exp(self):
        out_data = np.exp(self.data)
        def backward(g):
            if self.requires_grad:
                self._accumulate(g * out_data)
        return Tensor._make(out_data, (self,), backward, "exp")

    def log(self):
        def backward(g):
            if self.requires_grad:
                self._accumulate(g / self.data)
        return Tensor._make(np.log(self.data), (self,), backward, "log")

    def sqrt(self):
        return self ** 0.5

    def tanh(self):
        out_data = np.tanh(self.data)
        def backward(g):
            if self.requires_grad:
                self._accumulate(g * (1 - out_data ** 2))
        return Tensor._make(out_data, (self,), backward, "tanh")

    def sigmoid(self):
        from ..utils import sigmoid as _sig
        out_data = _sig(self.data)
        def backward(g):
            if self.requires_grad:
                self._accumulate(g * out_data * (1 - out_data))
        return Tensor._make(out_data, (self,), backward, "sigmoid")

    def relu(self):
        mask = self.data > 0
        def backward(g):
            if self.requires_grad:
                self._accumulate(g * mask)
        return Tensor._make(self.data * mask, (self,), backward, "relu")

    def abs(self):
        sign = np.sign(self.data)
        def backward(g):
            if self.requires_grad:
                self._accumulate(g * sign)
        return Tensor._make(np.abs(self.data), (self,), backward, "abs")

    def clip(self, lo, hi):
        mask = (self.data >= lo) & (self.data <= hi)
        def backward(g):
            if self.requires_grad:
                self._accumulate(g * mask)
        return Tensor._make(np.clip(self.data, lo, hi), (self,), backward, "clip")

    # ------------------------------------------------------------------
    # reductions
    # ------------------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        def backward(g):
            if not self.requires_grad:
                return
            if axis is None:
                self._accumulate(np.broadcast_to(g, self.data.shape))
            else:
                if not keepdims:
                    g = np.expand_dims(g, axis)
                self._accumulate(np.broadcast_to(g, self.data.shape))
        return Tensor._make(self.data.sum(axis=axis, keepdims=keepdims),
                            (self,), backward, "sum")

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            n = int(np.prod([self.data.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def var(self, axis=None, keepdims=False):
        mu = self.mean(axis=axis, keepdims=True)
        out = ((self - mu) ** 2).mean(axis=axis, keepdims=keepdims)
        return out

    def max(self, axis=None, keepdims=False):
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        def backward(g):
            if not self.requires_grad:
                return
            if axis is None:
                mask = (self.data == out_data)
                self._accumulate(g * mask / mask.sum())
            else:
                expanded = out_data if keepdims else np.expand_dims(out_data, axis)
                mask = (self.data == expanded)
                gexp = g if keepdims else np.expand_dims(g, axis)
                counts = mask.sum(axis=axis, keepdims=True)
                self._accumulate(mask * gexp / counts)
        return Tensor._make(out_data, (self,), backward, "max")

    def min(self, axis=None, keepdims=False):
        return -((-self).max(axis=axis, keepdims=keepdims))

    # ------------------------------------------------------------------
    # shape ops
    # ------------------------------------------------------------------
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        orig = self.data.shape
        def backward(g):
            if self.requires_grad:
                self._accumulate(g.reshape(orig))
        return Tensor._make(self.data.reshape(shape), (self,), backward, "reshape")

    def transpose(self, *axes):
        if not axes:
            axes = tuple(reversed(range(self.data.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        inv = np.argsort(axes)
        def backward(g):
            if self.requires_grad:
                self._accumulate(g.transpose(inv))
        return Tensor._make(self.data.transpose(axes), (self,), backward, "transpose")

    def swapaxes(self, a, b):
        axes = list(range(self.data.ndim))
        axes[a], axes[b] = axes[b], axes[a]
        return self.transpose(*axes)

    def __getitem__(self, idx):
        def backward(g):
            if self.requires_grad:
                full = np.zeros_like(self.data)
                np.add.at(full, idx, g)
                self._accumulate(full)
        return Tensor._make(self.data[idx], (self,), backward, "getitem")

    def pad(self, pad_width, value=0.0):
        slices = tuple(slice(lo, s + lo) for (lo, hi), s in zip(pad_width, self.data.shape))
        def backward(g):
            if self.requires_grad:
                self._accumulate(g[slices])
        return Tensor._make(np.pad(self.data, pad_width, constant_values=value),
                            (self,), backward, "pad")

    @staticmethod
    def concatenate(tensors, axis=0):
        tensors = [Tensor._wrap(t) for t in tensors]
        sizes = [t.data.shape[axis] for t in tensors]
        offsets = np.cumsum([0] + sizes)
        def backward(g):
            for t, lo, hi in zip(tensors, offsets[:-1], offsets[1:]):
                if t.requires_grad:
                    sl = [slice(None)] * g.ndim
                    sl[axis] = slice(lo, hi)
                    t._accumulate(g[tuple(sl)])
        return Tensor._make(np.concatenate([t.data for t in tensors], axis=axis),
                            tensors, backward, "concat")

    @staticmethod
    def stack(tensors, axis=0):
        tensors = [Tensor._wrap(t) for t in tensors]
        def backward(g):
            gs = np.moveaxis(g, axis, 0)
            for t, gi in zip(tensors, gs):
                if t.requires_grad:
                    t._accumulate(gi)
        return Tensor._make(np.stack([t.data for t in tensors], axis=axis),
                            tensors, backward, "stack")

    @staticmethod
    def where(cond, a, b):
        a, b = Tensor._wrap(a), Tensor._wrap(b)
        cond = np.asarray(cond, dtype=bool)
        def backward(g):
            if a.requires_grad:
                a._accumulate(np.where(cond, g, 0.0))
            if b.requires_grad:
                b._accumulate(np.where(cond, 0.0, g))
        return Tensor._make(np.where(cond, a.data, b.data), (a, b), backward, "where")


__all__ = ["Tensor", "no_grad", "is_grad_enabled"]
