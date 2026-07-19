"""Barycentric transport MAP: where does each source point go in the target?"""
import numpy as np
from ..utils import check_array, check_random_state


def _cost_matrix(X, Y, p=2):
    diff = X[:, None, :] - Y[None, :, :]
    return (np.abs(diff) ** p).sum(axis=2)


def ot_mapping(Xs, Xt, reg=0.1, p=2):
    """Barycentric transport MAP: where does each source point go in the target?

    OT gives a coupling (a soft matching); to actually MOVE a source point you need
    a map. The barycentric projection sends each source point to the weighted
    average of the target points it is coupled to::

        map(x_i) = sum_j T[i,j] * x_t[j] / sum_j T[i,j]

    This is the workhorse of OT domain adaptation -- transport the source features
    onto the target distribution, then train there. Returns the mapped source
    points (same shape as ``Xs``).
    """
    from .core import sinkhorn
    Xs = check_array(Xs); Xt = check_array(Xt)
    a = np.full(len(Xs), 1.0 / len(Xs))
    b = np.full(len(Xt), 1.0 / len(Xt))
    T = sinkhorn(a, b, _cost_matrix(Xs, Xt, p), reg=reg)[0]
    row = T.sum(axis=1, keepdims=True) + 1e-300
    return (T @ Xt) / row


__all__ = ["ot_mapping"]
