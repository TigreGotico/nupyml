"""Second-order steps you can TRUST, by bounding their size (Steihaug, 1983)."""
import numpy as np


def _steihaug_cg(g, Hp, radius, tol=1e-10):
    # truncated CG for the trust-region subproblem min g.p + 0.5 p.Hp, ||p||<=radius
    p = np.zeros_like(g); r = g.copy(); d = -g
    if np.linalg.norm(r) < tol:
        return p
    for _ in range(len(g) + 5):
        Hd = Hp(d)
        dHd = d @ Hd
        if dHd <= 0:                                     # negative curvature: go to boundary
            return _to_boundary(p, d, radius)
        alpha = (r @ r) / dHd
        p_new = p + alpha * d
        if np.linalg.norm(p_new) >= radius:
            return _to_boundary(p, d, radius)
        r_new = r + alpha * Hd
        if np.linalg.norm(r_new) < tol:
            return p_new
        beta = (r_new @ r_new) / (r @ r)
        d = -r_new + beta * d
        p, r = p_new, r_new
    return p


def _to_boundary(p, d, radius):
    # solve ||p + tau d|| = radius for tau >= 0
    a = d @ d; b = 2 * (p @ d); c = p @ p - radius ** 2
    tau = (-b + np.sqrt(max(b ** 2 - 4 * a * c, 0))) / (2 * a + 1e-16)
    return p + tau * d


def trust_region_newton_cg(f, grad, hessp, x0, max_iter=100, tol=1e-6,
                           max_radius=10.0):
    """Second-order steps you can TRUST, by bounding their size (Steihaug, 1983).

    Newton's method solves ``H p = -g`` and steps by ``p`` -- but far from the
    optimum the quadratic model is unreliable and that step can overshoot or, if
    ``H`` is indefinite, point uphill. Trust-region methods only trust the model
    within a RADIUS: they minimise it subject to ``||p|| <= radius`` (via truncated
    CG, which also handles negative curvature gracefully), then GROW the radius when
    the model predicted the actual decrease well and SHRINK it when it did not. The
    result is Newton's fast local convergence with global robustness. ``hessp(x, v)``
    returns the Hessian-vector product.
    """
    x = np.asarray(x0, float).copy()
    radius = 1.0
    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        p = _steihaug_cg(g, lambda v: hessp(x, v), radius)
        actual = f(x) - f(x + p)
        pred = -(g @ p + 0.5 * p @ hessp(x, p))
        rho = actual / (pred + 1e-16)
        if rho < 0.25:
            radius *= 0.25
        elif rho > 0.75 and np.linalg.norm(p) >= 0.99 * radius:
            radius = min(2 * radius, max_radius)
        if rho > 0.1:                                    # accept the step
            x = x + p
    return x


__all__ = ["trust_region_newton_cg"]
