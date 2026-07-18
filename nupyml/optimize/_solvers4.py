"""Numerical optimization v4: constrained SQP, multipliers, variance-reduced
stochastic gradients, mirror descent, and proximal Newton.

Five more optimizers. SLSQP solves smooth EQUALITY-constrained problems by a sequence
of quadratic programs. The augmented Lagrangian turns constraints into a penalty that
is driven exact by updating multipliers. SAGA and SVRG give SGD the convergence of
full-batch gradient descent on finite sums. Mirror descent respects a geometry like
the probability simplex. Proximal Newton adds second-order curvature to an L1 problem.
"""
import numpy as np


def slsqp(f, grad, x0, eq_constraint, eq_jac, max_iter=100, tol=1e-8):
    """Solve an equality-constrained problem by a sequence of QPs (Kraft, 1988).

    Minimise ``f(x)`` subject to ``c(x) = 0``. SLSQP works like Newton's method for the
    KKT conditions: at each iterate it builds a QUADRATIC model of the Lagrangian and a
    LINEAR model of the constraints, and solves that equality-constrained QP for the
    step by its KKT linear system. A BFGS update keeps a positive-definite Hessian
    approximation so the QP is well posed without computing second derivatives. It is
    scipy's default for smooth constrained optimisation. ``eq_constraint(x)`` returns
    the constraint residual vector and ``eq_jac(x)`` its Jacobian.
    """
    x = np.asarray(x0, float).copy()
    n = len(x)
    B = np.eye(n)
    g = grad(x)
    for _ in range(max_iter):
        c = np.atleast_1d(eq_constraint(x))
        J = np.atleast_2d(eq_jac(x))
        m = len(c)
        # KKT system for min 0.5 d'B d + g'd  s.t.  J d = -c
        KKT = np.block([[B, J.T], [J, np.zeros((m, m))]])
        rhs = np.concatenate([-g, -c])
        sol = np.linalg.solve(KKT, rhs)
        d = sol[:n]
        if np.linalg.norm(d) < tol:
            break
        x_new = x + d
        g_new = grad(x_new)
        s, yv = d, g_new - g                              # BFGS on the Lagrangian curvature
        if s @ yv > 1e-10:
            Bs = B @ s
            B = B - np.outer(Bs, Bs) / (s @ Bs) + np.outer(yv, yv) / (yv @ s)
        x, g = x_new, g_new
    return x


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


def svrg(grad_i, x0, n_samples, lr=0.05, epochs=50, random_state=None):
    """SGD with the convergence of full-batch, via a SNAPSHOT (Johnson & Zhang, 2013).

    Plain SGD's gradient is noisy, so it must use a decaying step and converges slowly.
    Stochastic Variance-Reduced Gradient fixes this: periodically it computes the FULL
    gradient at a snapshot point, then each cheap step uses ``g_i(x) - g_i(snapshot) +
    full_grad`` -- an unbiased gradient whose variance SHRINKS to zero as ``x``
    approaches the optimum, because the correction cancels the noise. The result is a
    CONSTANT step size and LINEAR convergence on strongly-convex finite sums.
    ``grad_i(x, i)`` is the gradient of term ``i``.
    """
    rng = np.random.RandomState(random_state) if not isinstance(
        random_state, np.random.RandomState) else random_state
    x = np.asarray(x0, float).copy()
    for _ in range(epochs):
        snapshot = x.copy()
        full = np.mean([grad_i(snapshot, i) for i in range(n_samples)], axis=0)
        for _ in range(n_samples):
            i = rng.randint(n_samples)
            x = x - lr * (grad_i(x, i) - grad_i(snapshot, i) + full)
    return x


def saga(grad_i, x0, n_samples, lr=0.05, epochs=50, random_state=None):
    """Variance reduction with a gradient TABLE, no snapshots (Defazio, 2014).

    SVRG must periodically recompute a full gradient; SAGA avoids that by keeping a
    TABLE of the most recent gradient seen for each term. Each step picks a term,
    forms the corrected gradient ``g_i(x) - table_i + mean(table)``, updates ``x``,
    and refreshes ``table_i`` -- so the variance reduction is continuous, with no
    epochs of full-gradient computation. Same fast convergence as SVRG, and it also
    supports composite (proximal) objectives naturally. ``grad_i(x, i)`` is term
    ``i``'s gradient.
    """
    rng = np.random.RandomState(random_state) if not isinstance(
        random_state, np.random.RandomState) else random_state
    x = np.asarray(x0, float).copy()
    table = np.array([grad_i(x, i) for i in range(n_samples)])
    mean = table.mean(axis=0)
    for _ in range(epochs * n_samples):
        i = rng.randint(n_samples)
        gi = grad_i(x, i)
        x = x - lr * (gi - table[i] + mean)              # variance-reduced step
        mean += (gi - table[i]) / n_samples              # keep the running mean
        table[i] = gi
    return x


def mirror_descent(grad, x0, eta=0.1, max_iter=500, tol=1e-9):
    """Gradient descent in the RIGHT geometry -- here the simplex (Nemirovski, 1983).

    Ordinary gradient descent assumes a Euclidean space, which is wrong when the
    variable lives on the probability SIMPLEX (weights that must be non-negative and
    sum to one). Mirror descent replaces the Euclidean step with one measured by a
    "mirror map" -- for the simplex, the negative entropy -- yielding the EXPONENTIATED
    GRADIENT update ``x *= exp(-eta·grad); x /= sum(x)``. It stays on the simplex
    automatically and its convergence depends on the geometry's diameter, not the
    ambient dimension, which is why it excels in high-dimensional online learning.
    Starts from ``x0`` on the simplex.
    """
    x = np.asarray(x0, float).copy()
    x = x / x.sum()
    for _ in range(max_iter):
        g = grad(x)
        x_new = x * np.exp(-eta * g)                      # entropic mirror step
        x_new /= x_new.sum()
        if np.linalg.norm(x_new - x) < tol:
            x = x_new; break
        x = x_new
    return x


def proximal_newton(smooth_grad, smooth_hess, x0, l1=0.1, max_iter=100, tol=1e-8,
                    inner_iter=200):
    """Second-order curvature on the smooth part of an L1 problem (Lee, 2014).

    Proximal GRADIENT (ISTA/FISTA) uses only the gradient of the smooth loss, so it
    inherits gradient descent's slow convergence on ill-conditioned problems. Proximal
    NEWTON scales the step by the smooth part's HESSIAN: at each iterate it minimises a
    second-order model of the loss plus the exact L1 term (a lasso subproblem, solved
    by coordinate descent), giving Newton-fast convergence to a sparse solution. Ideal
    when the Hessian is available and the problem is badly conditioned. ``smooth_grad``
    and ``smooth_hess`` are of the SMOOTH loss only; the L1 term is handled internally.
    """
    x = np.asarray(x0, float).copy()
    n = len(x)
    for _ in range(max_iter):
        g = smooth_grad(x)
        H = np.atleast_2d(smooth_hess(x))
        # solve min 0.5 (z-x)'H(z-x) + g'(z-x) + l1||z||_1 by coordinate descent
        z = x.copy()
        Hd = np.diag(H)
        for _ in range(inner_iter):
            for j in range(n):
                # gradient of the quadratic model wrt z_j, excluding z_j's own term
                rj = g[j] + H[j] @ (z - x) - Hd[j] * (z[j] - x[j])
                z[j] = _soft(x[j] - rj / (Hd[j] + 1e-12), l1 / (Hd[j] + 1e-12))
        if np.linalg.norm(z - x) < tol:
            x = z; break
        x = z
    return x


def _soft(v, thr):
    return np.sign(v) * max(abs(v) - thr, 0.0)


__all__ = ["slsqp", "augmented_lagrangian", "svrg", "saga", "mirror_descent",
           "proximal_newton"]
