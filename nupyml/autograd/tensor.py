"""Reverse-mode automatic differentiation over numpy arrays.

WHAT AUTODIFF IS, AND WHAT IT IS NOT
------------------------------------
It is not symbolic differentiation: nothing here manipulates formulas or
returns an expression for the derivative. It is not numerical differentiation
either: nothing is estimated by nudging inputs and measuring the change (that
would cost one forward pass per parameter, and lose precision to round-off).

Autodiff is the observation that any program built from differentiable
primitives is a composition of simple functions, and the chain rule turns the
derivative of a composition into a product of the pieces' derivatives. So if
every primitive knows its own local derivative, the derivative of any program
made of them follows mechanically.

FORWARD MODE VS REVERSE MODE
----------------------------
The chain rule for ``f(g(h(x)))`` gives a product of Jacobians::

    df/dx = J_f @ J_g @ J_h

Matrix products are associative, so we may multiply left-to-right or
right-to-left, and the choice decides the cost.

- *Forward mode* goes right-to-left, propagating one input's influence
  forward. One pass gives the derivative of EVERY output with respect to ONE
  input.
- *Reverse mode* goes left-to-right, propagating one output's sensitivity
  backward. One pass gives the derivative of ONE output with respect to EVERY
  input.

Machine learning almost always minimises a single scalar loss over millions of
parameters, so reverse mode wins overwhelmingly: one backward pass yields every
gradient. That single scalar output is why it is called backpropagation.

The price is memory. Reverse mode has to know the graph before it can walk it
backwards, so intermediate values are kept alive until the backward pass uses
them. Forward mode needs no such tape.

HOW THIS FILE DOES IT
---------------------
Every operation on a ``Tensor`` does two things:

1. computes the output value immediately (this is "define-by-run": the graph is
   a side effect of running the code, not a structure declared up front); and
2. records a closure that knows how to convert a gradient flowing INTO the
   output into gradients flowing OUT to each input.

That closure is a vector-Jacobian product, or VJP. It never materialises the
Jacobian -- for ``y = x @ W`` with a 1000x1000 W, the Jacobian would have a
trillion entries, while the VJP is just another matmul.

``backward()`` then walks the recorded graph in reverse topological order and
calls each closure. The ordering is not a detail; see ``Tensor.backward``.
"""
import contextlib

import numpy as np

# Tracking is global rather than per-tensor so that `no_grad()` can switch off
# an entire region of code, including library calls it does not control.
_grad_enabled = True


@contextlib.contextmanager
def no_grad():
    """Run a block without recording a graph.

    Inference needs values, not gradients. Recording anyway would keep every
    intermediate alive for a backward pass that never comes -- pure waste, and
    on a long evaluation loop it is the difference between constant and
    ever-growing memory.
    """
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
    """Undo broadcasting on the way back.

    Broadcasting silently copies data: adding a ``(3,)`` bias to a ``(64, 3)``
    batch reuses each bias element 64 times. Reuse means that on the backward
    pass, the bias receives 64 separate contributions -- and the chain rule says
    contributions to the same variable ADD.

    So the adjoint of "copy" is "sum". Wherever the forward pass stretched a
    dimension, the backward pass sums over it, which also restores the original
    shape. Getting this wrong is the classic autodiff bug: the gradient is the
    right numbers in the wrong shape, and numpy broadcasts it again instead of
    complaining.
    """
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
    """An ndarray that remembers how it was computed.

    Attributes
    ----------
    data : ndarray
        The value. Always float64 here: autodiff on integers is meaningless,
        and float32 would make gradient checking against finite differences
        hopelessly noisy.
    grad : ndarray or None
        Filled in by ``backward()``. ``None`` means "no gradient has arrived",
        which is deliberately distinct from "the gradient is zero".
    requires_grad : bool
        Whether this tensor is part of the graph. Inputs and constants are
        ``False``; parameters are ``True``. It spreads: any result computed
        from a tracked tensor is itself tracked.
    _backward : callable or None
        The VJP closure. Takes the gradient arriving at this node, and adds the
        appropriate share to each input's ``.grad``.
    _prev : tuple of Tensor
        The inputs this node was computed from -- the graph edges.
    """

    # __slots__ rather than a dict: a graph holds one of these per intermediate
    # value, so per-object overhead is multiplied by the size of the network.
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")
    # tells numpy to defer to our __radd__/__rmul__ instead of trying to
    # broadcast a Tensor into an ndarray elementwise
    __array_priority__ = 100

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
        """Build a result node, recording the graph only if it is needed.

        A node joins the graph only when at least one input is tracked. This is
        what stops a forward pass through frozen weights, or any call inside
        ``no_grad()``, from paying for bookkeeping it will never use: the
        closure and the parent references are simply never stored, and the
        intermediates become garbage immediately.
        """
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
        """A view of the same values with the history cut off.

        Gradients stop here. Useful whenever a value should be treated as a
        constant even though it was computed from parameters -- a target, or a
        quantity you deliberately do not want to backpropagate through.
        """
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self):
        self.grad = None

    # ------------------------------------------------------------------
    # backward
    # ------------------------------------------------------------------
    def backward(self, grad=None):
        """Propagate gradients from this tensor back to every leaf.

        WHY TOPOLOGICAL ORDER
        ---------------------
        A node's gradient is the SUM of contributions from every path that
        leads out of it. If a tensor is used twice -- ``y = x*x + x`` -- then x
        gets two contributions, and applying x's own VJP before both have
        arrived would propagate a half-finished gradient onward. Wrong answer,
        silently.

        So the rule is: never process a node until every node that depends on it
        has been processed. That is exactly reverse topological order. It is
        also why the seed gradient starts at 1: d(loss)/d(loss) = 1, and every
        other gradient is that seed routed backwards.

        The traversal below is iterative rather than recursive, because a deep
        network (say an RNN unrolled over a long sequence) would otherwise
        blow Python's stack.

        Parameters
        ----------
        grad : ndarray, optional
            The gradient arriving at this tensor. Defaults to ones for a
            scalar. A non-scalar has no canonical seed -- "the derivative of a
            vector" is a Jacobian, and reverse mode computes one row at a time
            -- so the caller must say which combination it wants.
        """
        if not self.requires_grad:
            raise RuntimeError("Called backward on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensors")
            grad = np.ones_like(self.data)
        else:
            grad = np.asarray(grad, dtype=np.float64)

        # depth-first post-order gives reverse topological order when reversed.
        # the `processed` flag emulates the "visit children, then me" moment
        # that recursion gets for free.
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
        """Add an incoming contribution to this tensor's gradient.

        Accumulate, never overwrite: a tensor feeding several operations
        receives one contribution per use, and the chain rule sums them. This
        is also why training loops must call ``zero_grad()`` -- otherwise the
        next step's gradients pile on top of the last step's.
        """
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
        """Matrix product, the workhorse of every dense layer.

        For ``C = A @ B`` the VJPs are::

            dL/dA = dL/dC @ B'
            dL/dB = A' @ dL/dC

        Worth seeing why rather than memorising: ``C_ij = sum_k A_ik B_kj``, so
        ``A_ik`` influences the whole i-th row of C, weighted by row k of B.
        Summing those influences is precisely ``dL/dC @ B'``. The transposes are
        not a trick -- they are what "sum over the shared index" looks like.

        This is also the clearest case for VJPs over Jacobians: the true
        Jacobian here is a four-dimensional object, while the gradient is two
        ordinary matmuls.
        """
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
        # d/dx exp(x) = exp(x): the output is captured by the closure and
        # reused, rather than recomputed on the backward pass. Trading memory
        # for arithmetic like this is the whole reason a tape exists.
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
        """Rectifier: ``max(x, 0)``.

        The derivative is 1 where the input was positive and 0 elsewhere -- so
        backward is just a mask. Two consequences worth knowing:

        * Gradients pass through the active half UNSCALED. Sigmoid and tanh
          saturate and multiply gradients by something well under 1 at every
          layer, which is what made deep networks untrainable for years. ReLU
          not shrinking the gradient is most of why depth became practical.
        * The dead half passes nothing. A unit pushed negative for every input
          receives no gradient and can never recover -- a "dead" ReLU. That is
          what leaky variants exist to avoid.

        Strictly, ``max(x, 0)`` has no derivative at exactly 0. Sub-gradient
        conventions allow anything in [0, 1]; the mask picks 0, and an exact
        zero input is a measure-zero event anyway.
        """
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
        """Sum, whose adjoint is broadcast.

        Summing collapses many inputs into one output, and each contributed
        with weight 1 -- so the gradient arriving at the output is copied back
        to every element that fed it. Note the duality with ``_unbroadcast``:
        the adjoint of sum is broadcast, and the adjoint of broadcast is sum.
        """
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
        """Maximum, with the gradient routed only to the winner.

        ``max`` is a selector: the output IS one of the inputs, so the gradient
        flows entirely to whichever element won and not at all to the others.
        This is what makes max-pooling backward a scatter.

        Ties are split evenly between the winners. Any single winner would also
        be a valid sub-gradient; splitting keeps the result independent of
        argmax's arbitrary tie-breaking, so the same input always gives the
        same gradient.
        """
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
