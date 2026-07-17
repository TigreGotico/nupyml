"""From-scratch mathematical optimization -- the solvers under half of ML.

Most estimators hand-roll their own optimisation; this package makes the standard
methods explicit and teachable: linear and quadratic programming, proximal
methods for non-smooth (L1) objectives, a Krylov solver for SPD systems, and
three derivative-free searches for when you have no gradient at all.
"""
import numpy as np


# --- linear programming ---------------------------------------------------

def linprog_simplex(c, A_ub, b_ub, max_iter=1000):
    """Minimise ``c·x`` subject to ``A_ub x <= b_ub`` and ``x >= 0`` (b_ub >= 0).

    THE SIMPLEX IDEA
    ----------------
    The feasible region of an LP is a polytope, and the optimum (if finite) sits
    at a VERTEX. Simplex walks from vertex to adjacent vertex, each step along an
    edge that lowers the objective, until no improving edge remains. Mechanically
    it pivots a tableau: pick an entering variable with negative reduced cost
    (Bland's rule here, to avoid cycling), a leaving variable by the min-ratio
    test (the first constraint that binds), and pivot. Slack variables turn the
    ``<=`` rows into equalities, and ``x = 0`` is the starting vertex.

    Returns (x, optimal_value). Handles the standard ``<=``, non-negative form;
    equality constraints would need a phase-1 to find a starting vertex.
    """
    c = np.asarray(c, float)
    A = np.asarray(A_ub, float)
    b = np.asarray(b_ub, float)
    m, n = A.shape
    # tableau: [A | I | b] on top, [c | 0 | 0] as the objective row
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = A
    T[:m, n:n + m] = np.eye(m)
    T[:m, -1] = b
    T[-1, :n] = c
    basis = list(range(n, n + m))                    # start with slacks basic
    for _ in range(max_iter):
        # Bland's rule: lowest-index column with negative reduced cost enters
        cols = np.where(T[-1, :-1] < -1e-9)[0]
        if len(cols) == 0:
            break                                    # optimal
        j = cols[0]
        ratios = np.where(T[:m, j] > 1e-9, T[:m, -1] / T[:m, j], np.inf)
        if np.all(~np.isfinite(ratios)):
            raise ValueError("LP is unbounded")
        i = int(np.argmin(ratios))
        T[i] /= T[i, j]                              # pivot
        for k in range(m + 1):
            if k != i:
                T[k] -= T[k, j] * T[i]
        basis[i] = j
    x = np.zeros(n + m)
    for i, bvar in enumerate(basis):
        x[bvar] = T[i, -1]
    return x[:n], float(c @ x[:n])


# --- quadratic programming -------------------------------------------------

def quadratic_program(P, q, G=None, h=None, max_iter=100):
    """Minimise ``0.5 x'Px + q'x`` s.t. ``Gx <= h`` by the ACTIVE-SET method.

    THE ACTIVE-SET IDEA
    -------------------
    At the solution some inequality constraints hold with EQUALITY (active) and
    the rest are slack. If you knew which, the problem is a simple equality-
    constrained QP with a closed-form KKT solution. You don't, so active-set
    guesses a working set and repairs it: solve the equality QP on the current
    set; if the step is blocked by an inactive constraint, ADD it; if a
    constraint's Lagrange multiplier goes negative (it is pulling the wrong way),
    DROP it. Each move strictly helps, so it converges in finitely many steps for
    a strictly convex ``P``.

    Requires ``P`` positive definite and a feasible start (``x=0`` when ``h>=0``).
    Returns the minimiser.
    """
    P = np.asarray(P, float)
    q = np.asarray(q, float)
    n = len(q)
    if G is None:
        return np.linalg.solve(P, -q)                # unconstrained
    G = np.asarray(G, float)
    h = np.asarray(h, float)
    x = np.zeros(n)
    if np.any(G @ x > h + 1e-9):
        raise ValueError("x=0 infeasible; provide a feasible start")
    working = [i for i in range(len(h)) if abs(G[i] @ x - h[i]) < 1e-9]
    for _ in range(max_iter):
        g = P @ x + q
        Gw = G[working] if working else np.zeros((0, n))
        # solve KKT for the step p that minimises the QP on the working set
        if len(working):
            KKT = np.block([[P, Gw.T], [Gw, np.zeros((len(working),) * 2)]])
            rhs = np.concatenate([-g, np.zeros(len(working))])
            sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
            p, lam = sol[:n], sol[n:]
        else:
            p, lam = np.linalg.solve(P, -g), np.array([])
        if np.linalg.norm(p) < 1e-8:
            if len(lam) == 0 or np.all(lam >= -1e-9):
                return x                             # KKT satisfied -> optimum
            j = working[int(np.argmin(lam))]         # drop wrong-sign constraint
            working.remove(j)
        else:
            # largest step in [0,1] before an inactive constraint blocks
            alpha, block = 1.0, None
            for i in range(len(h)):
                if i in working:
                    continue
                denom = G[i] @ p
                if denom > 1e-12:
                    t = (h[i] - G[i] @ x) / denom
                    if t < alpha:
                        alpha, block = t, i
            x = x + alpha * p
            if block is not None:
                working.append(block)
    return x


# --- iterative linear solve ------------------------------------------------

def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    """Solve ``A x = b`` for symmetric positive-definite ``A`` (Hestenes-Stiefel).

    CG minimises the quadratic ``0.5 x'Ax - b'x`` (whose minimum solves the
    system) along a sequence of A-CONJUGATE directions, so each step's progress is
    never undone by the next -- unlike steepest descent, which zig-zags. It
    reaches the exact solution in at most ``n`` steps and, crucially, touches ``A``
    only through matrix-VECTOR products, so it scales to huge sparse systems where
    factorising ``A`` is impossible.
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    x = np.zeros_like(b) if x0 is None else np.array(x0, float)
    r = b - A @ x
    p = r.copy()
    rs = r @ r
    for _ in range(max_iter or len(b)):
        Ap = A @ p
        alpha = rs / (p @ Ap + 1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs) * p                    # conjugate direction update
        rs = rs_new
    return x


# --- proximal methods ------------------------------------------------------

def fista(grad_f, prox, x0, L, n_iter=200):
    """FISTA: accelerated proximal gradient for ``min f(x) + g(x)`` (Beck-Teboulle).

    When the objective splits into a SMOOTH ``f`` (you have its gradient) and a
    non-smooth ``g`` (you have its proximal operator -- e.g. soft-thresholding for
    an L1 penalty), proximal gradient alternates a gradient step on ``f`` with a
    prox step on ``g``. FISTA adds Nesterov MOMENTUM via an extrapolated point,
    lifting the convergence rate from ``O(1/k)`` to ``O(1/k^2)`` at no extra cost
    per iteration. ``L`` is a Lipschitz constant of ``grad_f`` (the step is 1/L).
    """
    x = np.array(x0, float)
    y = x.copy()
    t = 1.0
    for _ in range(n_iter):
        x_new = prox(y - grad_f(y) / L, 1.0 / L)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)  # momentum extrapolation
        x, t = x_new, t_new
    return x


def soft_threshold(x, thresh):
    """Prox of the L1 norm: shrink toward zero by ``thresh``, clamping at 0."""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def lasso_fista(X, y, alpha, n_iter=500):
    """Solve the lasso ``0.5||Xw - y||^2 + alpha||w||_1`` with FISTA."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    L = np.linalg.norm(X, 2) ** 2                    # Lipschitz const of the smooth part
    grad = lambda w: X.T @ (X @ w - y)
    prox = lambda w, s: soft_threshold(w, alpha * s)
    return fista(grad, prox, np.zeros(X.shape[1]), L, n_iter)


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

def nelder_mead(f, x0, step=0.5, max_iter=1000, tol=1e-8):
    """Nelder-Mead: minimise using only function VALUES, no gradient.

    Keeps a simplex of ``n+1`` points and reshapes it toward lower values by four
    moves -- reflect the worst point through the centroid, EXPAND if that is very
    good, CONTRACT if it is poor, and SHRINK the whole simplex if nothing helps.
    The go-to when the objective is noisy, non-differentiable, or a black box
    (hyperparameters, simulation outputs), though it has no convergence guarantee
    in high dimensions.
    """
    x0 = np.array(x0, float)
    n = len(x0)
    simplex = [x0] + [x0 + step * np.eye(n)[i] for i in range(n)]
    simplex = np.array(simplex)
    fvals = np.array([f(x) for x in simplex])
    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex, fvals = simplex[order], fvals[order]
        if np.std(fvals) < tol:
            break
        centroid = simplex[:-1].mean(axis=0)         # exclude the worst
        worst = simplex[-1]
        refl = centroid + (centroid - worst)         # reflection
        fr = f(refl)
        if fr < fvals[0]:
            exp = centroid + 2 * (centroid - worst)  # expansion
            fe = f(exp)
            simplex[-1], fvals[-1] = (exp, fe) if fe < fr else (refl, fr)
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = refl, fr
        else:
            cont = centroid + 0.5 * (worst - centroid)   # contraction
            fc = f(cont)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = cont, fc
            else:                                    # shrink toward the best
                simplex = simplex[0] + 0.5 * (simplex - simplex[0])
                fvals = np.array([f(x) for x in simplex])
    i = int(np.argmin(fvals))
    return simplex[i], float(fvals[i])


def simulated_annealing(f, x0, bounds=None, T0=1.0, cooling=0.995,
                        max_iter=5000, random_state=None):
    """Simulated annealing: escape local minima by accepting UPHILL moves early.

    Propose a random nearby point; always accept it if it is better, and accept it
    even if WORSE with probability ``exp(-Delta/T)``. The temperature ``T`` starts
    high (almost any move accepted -- broad exploration) and cools geometrically
    (only improving moves survive -- local refinement). The occasional uphill move
    is what lets it climb out of a local basin, unlike greedy descent. Inspired by
    annealing in metallurgy.
    """
    from ..utils import check_random_state
    rng = check_random_state(random_state)
    x = np.array(x0, float)
    fx = f(x)
    best, fbest = x.copy(), fx
    T = T0
    for _ in range(max_iter):
        cand = x + rng.normal(0, T + 0.05, size=x.shape)
        if bounds is not None:
            cand = np.clip(cand, bounds[0], bounds[1])
        fc = f(cand)
        if fc < fx or rng.rand() < np.exp(-(fc - fx) / max(T, 1e-12)):
            x, fx = cand, fc                         # accept (maybe uphill)
            if fc < fbest:
                best, fbest = cand.copy(), fc
        T *= cooling                                 # cool down
    return best, float(fbest)


def particle_swarm(f, bounds, n_particles=30, max_iter=200, w=0.7, c1=1.5,
                   c2=1.5, random_state=None):
    """Particle swarm: a population that steers toward good regions collectively.

    Each particle is a candidate flying through the space with a velocity pulled
    toward (a) the best point IT has seen and (b) the best point the WHOLE SWARM
    has seen, plus inertia. The social term shares discoveries so the swarm
    concentrates on promising regions, while inertia keeps exploring -- a
    gradient-free global search that handles multi-modal, non-differentiable
    objectives. ``bounds`` is ``(low, high)`` arrays.
    """
    from ..utils import check_random_state
    rng = check_random_state(random_state)
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    d = len(lo)
    X = rng.uniform(lo, hi, (n_particles, d))
    V = rng.uniform(-1, 1, (n_particles, d)) * (hi - lo)
    pbest = X.copy()
    pbest_f = np.array([f(x) for x in X])
    g = int(np.argmin(pbest_f))
    gbest, gbest_f = pbest[g].copy(), pbest_f[g]
    for _ in range(max_iter):
        r1, r2 = rng.rand(n_particles, d), rng.rand(n_particles, d)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, lo, hi)
        fx = np.array([f(x) for x in X])
        improved = fx < pbest_f
        pbest[improved], pbest_f[improved] = X[improved], fx[improved]
        g = int(np.argmin(pbest_f))
        if pbest_f[g] < gbest_f:
            gbest, gbest_f = pbest[g].copy(), pbest_f[g]
    return gbest, float(gbest_f)


__all__ = ["linprog_simplex", "quadratic_program", "conjugate_gradient",
           "fista", "soft_threshold", "lasso_fista", "admm_lasso",
           "nelder_mead", "simulated_annealing", "particle_swarm"]
