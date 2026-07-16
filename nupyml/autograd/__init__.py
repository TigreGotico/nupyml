"""Tape-based reverse-mode automatic differentiation over numpy arrays."""
from .tensor import Tensor, no_grad, is_grad_enabled
from . import functional
from .gradcheck import gradcheck

__all__ = ["Tensor", "no_grad", "is_grad_enabled", "functional", "gradcheck"]
