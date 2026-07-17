"""Automatic differentiation: gradients of any program built from these ops.

This is the engine under ``nupyml.nn``, and the shortest path to understanding
how neural network frameworks actually work.

* ``tensor.py``     -- the ``Tensor`` class and ``backward()``. Read the module
  docstring there for forward vs reverse mode, what a VJP is, and why the
  backward pass must run in reverse topological order.
* ``functional.py`` -- softmax, convolution, pooling and friends, hand-written
  rather than composed, for numerical stability and speed.
* ``gradcheck.py``  -- finite-difference verification. Every op in this package
  is checked against it in the test suite, which is what makes a hand-written
  derivative trustworthy.

The whole thing is a few hundred lines. The ideas are: record what you compute,
then walk it backwards.
"""
from .tensor import Tensor, no_grad, is_grad_enabled
from . import functional
from .gradcheck import gradcheck

__all__ = ["Tensor", "no_grad", "is_grad_enabled", "functional", "gradcheck"]
