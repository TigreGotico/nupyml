"""Prox of the L1 norm: shrink toward zero by ``thresh``, clamping at 0."""
import numpy as np


def soft_threshold(x, thresh):
    """Prox of the L1 norm: shrink toward zero by ``thresh``, clamping at 0."""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


__all__ = ["soft_threshold"]
