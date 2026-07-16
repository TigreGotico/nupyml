"""Finite-difference gradient checking."""
import numpy as np

from .tensor import Tensor


def gradcheck(fn, inputs, eps=1e-6, rtol=1e-4, atol=1e-6, raise_on_fail=True):
    """Check analytic gradients of ``fn(*inputs)`` against central differences.

    ``inputs`` is a sequence of Tensors with ``requires_grad=True``; ``fn``
    must return a Tensor of any shape (it is reduced with ``sum`` internally).
    """
    inputs = [t if isinstance(t, Tensor) else Tensor(t, requires_grad=True)
              for t in inputs]
    for t in inputs:
        t.zero_grad()
    out = fn(*inputs)
    out.sum().backward() if out.data.size > 1 else out.backward()
    analytic = [t.grad.copy() if t.grad is not None else np.zeros_like(t.data)
                for t in inputs]

    for t_idx, t in enumerate(inputs):
        numeric = np.zeros_like(t.data)
        flat = t.data.ravel()
        num_flat = numeric.ravel()
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + eps
            plus = fn(*inputs).data.sum()
            flat[i] = orig - eps
            minus = fn(*inputs).data.sum()
            flat[i] = orig
            num_flat[i] = (plus - minus) / (2 * eps)
        if not np.allclose(analytic[t_idx], numeric, rtol=rtol, atol=atol):
            if raise_on_fail:
                diff = np.abs(analytic[t_idx] - numeric).max()
                raise AssertionError(
                    f"Gradient check failed for input {t_idx}: max abs diff {diff:.3e}\n"
                    f"analytic:\n{analytic[t_idx]}\nnumeric:\n{numeric}"
                )
            return False
    return True


__all__ = ["gradcheck"]
