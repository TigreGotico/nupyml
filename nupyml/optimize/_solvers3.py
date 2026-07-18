"""Numerical optimization v3: trust-region, interior-point, stochastic-perturbation,
L1 quasi-Newton, and global search.

Five optimizers for cases the earlier solvers do not cover. Trust-region Newton-CG
takes reliable second-order steps by bounding their size. Interior-point solves
constrained QPs by walking down the middle of the feasible region. SPSA estimates a
gradient from two noisy function evaluations. OWL-QN extends L-BFGS to an L1
penalty. Basin-hopping escapes local minima by hopping between basins.
"""
import numpy as np

from .. utils import check_random_state


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


def interior_point_qp(P, q, G=None, h=None, max_iter=100, tol=1e-8):
    """Solve a constrained QP by walking down the MIDDLE of the feasible set
    (Karmarkar lineage).

    Minimise ``0.5 xᵀP x + qᵀx`` subject to ``G x <= h``. The active-set approach
    guesses which constraints bind; interior-point methods avoid the guess entirely.
    They add a LOG-BARRIER that pushes the iterate away from every constraint
    boundary, then relax the barrier toward zero while taking Newton steps on the
    perturbed optimality (KKT) conditions -- so the iterate glides through the strict
    interior and homes in on the solution, active constraints and all, in a
    predictable number of steps. Returns the optimal ``x``.
    """
    P = np.asarray(P, float); q = np.asarray(q, float)
    n = len(q)
    if G is None:
        return np.linalg.solve(P, -q)                    # unconstrained
    G = np.asarray(G, float); h = np.asarray(h, float)
    m = len(h)
    x = np.zeros(n)
    s = np.ones(m)                                       # slacks: h - Gx = s > 0
    lam = np.ones(m)                                     # multipliers
    # ensure a strictly feasible start
    while np.any(h - G @ x <= 0):
        x -= 0.1 * G.sum(axis=0)
    s = h - G @ x
    for _ in range(max_iter):
        mu = (s @ lam) / m
        if mu < tol:
            break
        t = 0.1 * mu                                     # barrier parameter
        # residuals of the perturbed KKT system
        r_dual = P @ x + q + G.T @ lam
        r_cent = s * lam - t
        Sinv = 1.0 / s
        # Schur complement on x
        H = P + G.T @ np.diag(lam * Sinv) @ G
        rhs = -(r_dual + G.T @ (Sinv * (r_cent - lam * (h - G @ x - s))))
        dx = np.linalg.solve(H + 1e-9 * np.eye(n), rhs)
        ds = -(G @ dx)                                   # from h - G x = s
        dlam = -Sinv * (r_cent + lam * ds)
        # step length keeping s, lam > 0
        step = 1.0
        for arr, darr in ((s, ds), (lam, dlam)):
            neg = darr < 0
            if np.any(neg):
                step = min(step, 0.99 * np.min(-arr[neg] / darr[neg]))
        x += step * dx; s += step * ds; lam += step * dlam
    return x


def spsa(f, x0, a=0.1, c=0.1, alpha=0.602, gamma=0.101, max_iter=1000,
         random_state=None):
    """Estimate a gradient from TWO function evaluations (Spall, 1992).

    Finite-difference gradients cost one evaluation per dimension -- hopeless in
    high dimensions or with noisy, expensive objectives. Simultaneous-Perturbation
    Stochastic Approximation perturbs ALL coordinates at once by a random ±vector and
    forms a gradient estimate from just TWO measurements, ``(f(x+cΔ) - f(x-cΔ)) / (2c)``
    divided component-wise by Δ. The estimate is noisy but UNBIASED for the true
    gradient's descent direction, so with decaying step sizes it converges -- at two
    evaluations per step regardless of dimension. Ideal for noisy simulation
    optimisation.
    """
    rng = check_random_state(random_state)
    x = np.asarray(x0, float).copy()
    for k in range(max_iter):
        ak = a / (k + 1 + 0.1 * max_iter) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=len(x))     # Rademacher perturbation
        ghat = (f(x + ck * delta) - f(x - ck * delta)) / (2 * ck) / delta
        x = x - ak * ghat
    return x


def owlqn(f, grad, x0, l1=0.1, m=10, max_iter=200, tol=1e-6):
    """L-BFGS extended to an L1 penalty (Andrew & Gao, 2007).

    L-BFGS needs a smooth objective, but the L1 penalty ``lambda||x||_1`` is
    non-differentiable at zero -- exactly where its useful sparsity happens.
    Orthant-Wise Limited-memory Quasi-Newton handles it by working within the ORTHANT
    (sign pattern) of the current point, where ``|x|`` IS smooth: it builds an
    L-BFGS direction from the smooth loss's gradient plus the pseudo-gradient of the
    penalty, projects the step to keep each coordinate in its orthant (so a
    coordinate crossing zero is clamped exactly TO zero), and line-searches. The
    result is genuinely sparse solutions with quasi-Newton speed. ``f``/``grad`` are
    the SMOOTH loss only; the L1 term is added internally.
    """
    x = np.asarray(x0, float).copy()
    s_list, y_list = [], []

    def pseudo_grad(x, g):
        pg = np.empty_like(x)
        for i in range(len(x)):
            if x[i] > 0:
                pg[i] = g[i] + l1
            elif x[i] < 0:
                pg[i] = g[i] - l1
            else:                                        # choose the steepest-descent side
                if g[i] + l1 < 0:
                    pg[i] = g[i] + l1
                elif g[i] - l1 > 0:
                    pg[i] = g[i] - l1
                else:
                    pg[i] = 0.0
        return pg

    for _ in range(max_iter):
        g = grad(x)
        pg = pseudo_grad(x, g)
        if np.linalg.norm(pg) < tol:
            break
        # L-BFGS two-loop on the pseudo-gradient
        q = pg.copy(); alphas = []
        for s, y in zip(reversed(s_list), reversed(y_list)):
            a = (s @ q) / (y @ s); alphas.append(a); q = q - a * y
        if y_list:
            q *= (s_list[-1] @ y_list[-1]) / (y_list[-1] @ y_list[-1])
        for s, y, a in zip(s_list, y_list, reversed(alphas)):
            q = q + (a - (y @ q) / (y @ s)) * s
        d = -q
        d[np.sign(d) != np.sign(-pg)] = 0.0              # align direction with -pg
        # backtracking line search on the full (loss + L1) objective, projecting
        # each trial point back into the starting orthant
        obj = lambda z: f(z) + l1 * np.abs(z).sum()
        step = 1.0; f0 = obj(x); xn = x
        for _ in range(30):
            xn = _orthant_project(x, x + step * d)
            if obj(xn) < f0:
                break
            step *= 0.5
        s = xn - x; y = grad(xn) - g
        if s @ y > 1e-10:
            if len(s_list) == m:
                s_list.pop(0); y_list.pop(0)
            s_list.append(s); y_list.append(y)
        x = xn
    return x


def _orthant_project(x_old, x_new):
    # clamp any coordinate that crossed zero back to exactly zero
    crossed = np.sign(x_new) != np.sign(x_old)
    keep_zero = crossed & (x_old != 0)
    out = x_new.copy()
    out[keep_zero] = 0.0
    return out


def basin_hopping(f, x0, step_size=1.0, n_iter=100, T=1.0, local=None,
                  random_state=None):
    """Hop between BASINS to escape local minima (Wales & Doye, 1997).

    Local optimisation only ever finds the nearest valley. Basin-hopping wraps it in
    a global search: from the current local minimum, take a random PERTURBATION,
    re-optimise to the bottom of whatever basin you land in, and accept the new
    minimum by a Metropolis rule (always if better, sometimes if worse, governed by
    ``T``). Because it explores the discrete set of local minima rather than the raw
    landscape, it is remarkably effective on rugged, many-minima problems (it was
    invented for molecular energy landscapes). ``local`` is a local minimiser
    ``x -> x_min`` (Nelder-Mead by default).
    """
    from ._solvers import nelder_mead
    rng = check_random_state(random_state)
    if local is None:
        local = lambda x: np.asarray(nelder_mead(f, x, max_iter=200)[0])
    x = local(np.asarray(x0, float))
    fx = f(x)
    best, best_f = x.copy(), fx
    for _ in range(n_iter):
        trial = local(x + rng.uniform(-step_size, step_size, size=len(x)))
        ft = f(trial)
        if ft < fx or rng.rand() < np.exp(-(ft - fx) / T):   # Metropolis accept
            x, fx = trial, ft
        if ft < best_f:
            best, best_f = trial.copy(), ft
    return best


__all__ = ["trust_region_newton_cg", "interior_point_qp", "spsa", "owlqn",
           "basin_hopping"]
