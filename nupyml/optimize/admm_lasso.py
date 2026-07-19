"""Solve the lasso by ADMM -- split the smooth fit from the L1 penalty."""
import numpy as np
from .soft_threshold import soft_threshold


def admm_lasso(X, y, alpha, rho=1.0, n_iter=200):
    """Solve the lasso by ADMM -- split the smooth fit from the L1 penalty.

    ADMM introduces a copy ``z`` of the variable, constrains ``x = z``, and
    alternates: an ``x``-update that is a ridge solve (smooth part), a ``z``-update
    that is soft-thresholding (the L1 part), and a dual update that drives ``x``
    and ``z`` together. Splitting a hard joint problem into two easy subproblems
    linked by a dual variable is ADMM's whole trick, and it parallelises and
    handles constraints that gradient methods cannot.
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    n_features = X.shape[1]
    XtX = X.T @ X
    Xty = X.T @ y
    inv = np.linalg.inv(XtX + rho * np.eye(n_features))
    x = np.zeros(n_features); z = np.zeros(n_features); u = np.zeros(n_features)
    for _ in range(n_iter):
        x = inv @ (Xty + rho * (z - u))              # ridge-like solve
        z = soft_threshold(x + u, alpha / rho)       # L1 prox
        u = u + x - z                                # dual ascent
    return z


# --- derivative-free -------------------------------------------------------


__all__ = ["admm_lasso"]
