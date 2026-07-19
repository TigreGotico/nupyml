"""Drive a constraint exact by updating MULTIPLIERS (Hestenes; Powell, 1969)."""
import numpy as np


def augmented_lagrangian(f, grad, x0, constraint, constraint_jac, rho0=1.0,
                         max_outer=30, max_inner=100, tol=1e-8):
    """Drive a constraint exact by updating MULTIPLIERS (Hestenes; Powell, 1969).

    A quadratic penalty ``rho||c(x)||^2`` only satisfies the constraint as ``rho ->
    inf``, which wrecks conditioning. The augmented Lagrangian adds LAGRANGE
    MULTIPLIERS to the penalty: it minimises ``f + lambda·c + rho/2||c||^2`` for the
    current multipliers, then updates ``lambda += rho·c``. The multiplier update makes
    the constraint exact at a MODERATE penalty, so it converges without the
    ill-conditioning of the pure penalty method. Equality constraints
    ``constraint(x) = 0`` here; inner minimisation by gradient descent.
    """
    x = np.asarray(x0, float).copy()
    lam = np.zeros(len(np.atleast_1d(constraint(x))))
    rho = rho0
    for _ in range(max_outer):
        for _ in range(max_inner):                        # minimise the augmented objective
            c = np.atleast_1d(constraint(x))
            J = np.atleast_2d(constraint_jac(x))
            g = grad(x) + J.T @ (lam + rho * c)
            if np.linalg.norm(g) < tol:
                break
            step = 1.0 / (1.0 + rho * np.sum(J ** 2))     # stability-scaled step
            x = x - step * g
        c = np.atleast_1d(constraint(x))
        lam = lam + rho * c                               # multiplier update
        if np.linalg.norm(c) < tol:
            break
        rho = min(rho * 2, 1e4)                           # tighten (bounded)
    return x


__all__ = ["augmented_lagrangian"]
