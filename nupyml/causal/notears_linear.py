"""NOTEARS: learn a causal DAG by CONTINUOUS optimization (Zheng et al., 2018)."""
import numpy as np
from ..utils import check_array


def notears_linear(X, lambda1=0.1, max_iter=100, h_tol=1e-8, rho_max=1e16):
    """NOTEARS: learn a causal DAG by CONTINUOUS optimization (Zheng et al., 2018).

    THE BREAKTHROUGH
    ----------------
    Learning a DAG was a combinatorial nightmare -- searching over acyclic
    structures is NP-hard. NOTEARS turned it into smooth optimization with one
    trick: a differentiable measure of "how cyclic" a weighted adjacency ``W`` is,
    ``h(W) = tr(e^{W∘W}) - d``, which is EXACTLY zero iff the graph is acyclic. So
    you minimise the reconstruction loss ``||X - XW||^2 + lambda1||W||_1`` subject
    to ``h(W)=0``, enforced by an augmented-Lagrangian outer loop. The result is a
    weighted DAG learned by gradient descent -- no discrete search.

    Returns the weighted adjacency ``W`` (``W[i,j] != 0`` means i -> j).
    """
    from scipy.linalg import expm
    from scipy.optimize import minimize
    X = check_array(X)
    n, d = X.shape

    def loss(w):
        W = w.reshape(d, d)
        R = X - X @ W
        f = 0.5 / n * (R ** 2).sum()
        G = -1.0 / n * X.T @ R
        return f, G.flatten()

    def h_func(W):
        E = expm(W * W)
        return np.trace(E) - d, E.T * W * 2           # value and gradient

    rho, alpha, w = 1.0, 0.0, np.zeros(d * d)
    for _ in range(max_iter):
        def obj(w):
            W = w.reshape(d, d)
            f, Gf = loss(w)
            h, Gh = h_func(W)
            total = f + 0.5 * rho * h * h + alpha * h + lambda1 * np.abs(w).sum()
            grad = Gf + (rho * h + alpha) * Gh.flatten() + lambda1 * np.sign(w)
            return total, grad
        sol = minimize(obj, w, jac=True, method="L-BFGS-B")
        w = sol.x
        h_val = h_func(w.reshape(d, d))[0]
        alpha += rho * h_val
        if h_val <= h_tol or rho >= rho_max:
            break
        rho *= 10
    W = w.reshape(d, d)
    W[np.abs(W) < 0.3] = 0                            # threshold weak edges
    return W


__all__ = ["notears_linear"]
