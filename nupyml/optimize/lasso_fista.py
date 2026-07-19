"""Solve the lasso ``0.5||Xw - y||^2 + alpha||w||_1`` with FISTA."""
import numpy as np
from .fista import fista
from .soft_threshold import soft_threshold


def lasso_fista(X, y, alpha, n_iter=500):
    """Solve the lasso ``0.5||Xw - y||^2 + alpha||w||_1`` with FISTA."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    L = np.linalg.norm(X, 2) ** 2                    # Lipschitz const of the smooth part
    grad = lambda w: X.T @ (X @ w - y)
    prox = lambda w, s: soft_threshold(w, alpha * s)
    return fista(grad, prox, np.zeros(X.shape[1]), L, n_iter)


__all__ = ["lasso_fista"]
