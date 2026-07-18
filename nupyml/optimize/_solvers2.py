"""Numerical optimization v2: quasi-Newton, nonlinear least squares, projection-
free, derivative-free, and sampling-based methods.

Five optimizers covering cases the gradient-descent/CG/FISTA core does not.
L-BFGS approximates the inverse Hessian from recent steps -- Newton's speed without
its cost. Levenberg-Marquardt is the standard for least-squares curve fitting.
Frank-Wolfe stays inside a constraint set without projecting. Powell needs no
gradient at all. The cross-entropy method optimizes by sampling and refitting a
distribution.
"""
import numpy as np


def lbfgs(f, grad, x0, m=10, max_iter=200, tol=1e-6, c1=1e-4, c2=0.9):
    """Newton-like steps from a LIMITED MEMORY of past gradients (Nocedal, 1980).

    Newton's method needs the inverse Hessian -- ``O(n^2)`` storage and ``O(n^3)``
    to invert, impossible for large ``n``. L-BFGS never forms it: it reconstructs
    the Hessian's ACTION on the gradient from just the last ``m`` steps and gradient
    changes (the two-loop recursion), giving curvature-aware steps with ``O(mn)``
    memory. It is the default batch optimizer for smooth problems (logistic
    regression, CRFs, many MLE fits) precisely because it scales. A Wolfe line
    search keeps each step stable.
    """
    x = np.asarray(x0, float).copy()
    g = grad(x)
    s_list, y_list, rho_list = [], [], []
    for it in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        q = g.copy()                                     # two-loop recursion
        alphas = []
        for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
            a = rho * (s @ q); alphas.append(a); q = q - a * y
        if y_list:
            gamma = (s_list[-1] @ y_list[-1]) / (y_list[-1] @ y_list[-1])
        else:
            gamma = 1.0
        z = gamma * q
        for s, y, rho, a in zip(s_list, y_list, rho_list, reversed(alphas)):
            beta = rho * (y @ z); z = z + (a - beta) * s
        d = -z                                           # search direction
        step = _wolfe_line_search(f, grad, x, d, g, c1, c2)
        x_new = x + step * d
        g_new = grad(x_new)
        s, y = x_new - x, g_new - g
        if s @ y > 1e-10:                                # keep positive curvature
            if len(s_list) == m:
                s_list.pop(0); y_list.pop(0); rho_list.pop(0)
            s_list.append(s); y_list.append(y); rho_list.append(1.0 / (s @ y))
        x, g = x_new, g_new
    return x


def _wolfe_line_search(f, grad, x, d, g, c1, c2, max_ls=25):
    step, lo, hi = 1.0, 0.0, np.inf
    f0 = f(x); gd0 = g @ d
    for _ in range(max_ls):
        xn = x + step * d
        if f(xn) > f0 + c1 * step * gd0:                 # Armijo (sufficient decrease)
            hi = step; step = 0.5 * (lo + hi)
        elif grad(xn) @ d < c2 * gd0:                    # curvature condition
            lo = step; step = 2 * lo if hi == np.inf else 0.5 * (lo + hi)
        else:
            return step
    return step


def levenberg_marquardt(residuals, jacobian, x0, max_iter=100, tol=1e-8,
                        lam0=1e-3):
    """Interpolate between gradient descent and Gauss-Newton (Levenberg 1944; ...).

    For a sum of squared residuals, Gauss-Newton is fast near the solution but can
    diverge far from it; gradient descent is safe but slow. Levenberg-Marquardt
    blends them with a damping ``lambda``: solve ``(JᵀJ + lambda·I) dx = -Jᵀr``.
    Large ``lambda`` -> a small, safe gradient step; small ``lambda`` -> the fast
    Gauss-Newton step. After a successful step ``lambda`` shrinks (trust the
    quadratic model more); after a failed one it grows. This adaptivity is why LM is
    the default for nonlinear least squares and curve fitting.
    """
    x = np.asarray(x0, float).copy()
    lam = lam0
    r = np.asarray(residuals(x), float)
    cost = r @ r
    for _ in range(max_iter):
        J = np.asarray(jacobian(x), float)
        JtJ = J.T @ J
        g = J.T @ r
        while True:
            try:
                dx = np.linalg.solve(JtJ + lam * np.eye(len(x)), -g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            r_new = np.asarray(residuals(x + dx), float)
            if r_new @ r_new < cost:                     # step accepted
                x = x + dx; r = r_new; cost = r @ r
                lam = max(lam / 10, 1e-12)               # trust the model more
                break
            lam *= 10                                    # step rejected: damp harder
            if lam > 1e12:
                return x
        if np.linalg.norm(g) < tol:
            break
    return x


def frank_wolfe(grad, linear_oracle, x0, max_iter=200, tol=1e-8):
    """Optimize over a constraint set WITHOUT ever projecting (Frank & Wolfe, 1956).

    Projected gradient needs a projection onto the feasible set, which can be as
    hard as the original problem. Frank-Wolfe (conditional gradient) instead calls a
    LINEAR oracle: minimise the current linear approximation over the set -- often a
    cheap closed form (a vertex of a simplex, the top singular vector of a nuclear-
    norm ball). It then steps toward that vertex by ``2/(t+2)``. Every iterate stays
    feasible as a convex combination of vertices, which also makes the solution
    SPARSE. ``linear_oracle(g)`` returns argmin over the set of ``g·s``.
    """
    x = np.asarray(x0, float).copy()
    for t in range(max_iter):
        g = grad(x)
        s = np.asarray(linear_oracle(g), float)          # linear minimisation oracle
        gap = g @ (x - s)                                # Frank-Wolfe duality gap
        if gap < tol:
            break
        gamma = 2.0 / (t + 2.0)                          # step toward the vertex
        x = x + gamma * (s - x)
    return x


def powell(f, x0, max_iter=100, tol=1e-8):
    """Minimise using only function VALUES -- no gradient (Powell, 1964).

    When the gradient is unavailable or unreliable (noisy simulations, black-box
    objectives), Powell's method searches along a set of directions, doing an exact
    1-D line minimisation along each, then REPLACES the set with the net direction
    of the whole sweep -- so the directions gradually align with the problem's
    valley and become conjugate, giving near-Newton progress without a single
    derivative. Directions start as the coordinate axes.
    """
    x = np.asarray(x0, float).copy()
    n = len(x)
    dirs = np.eye(n)
    fx = f(x)
    for _ in range(max_iter):
        x_start = x.copy(); f_start = fx
        biggest_drop = 0.0; drop_idx = 0
        for i in range(n):
            d = dirs[i]
            alpha = _line_min(f, x, d)
            new_fx = f(x + alpha * d)
            if fx - new_fx > biggest_drop:
                biggest_drop = fx - new_fx; drop_idx = i
            x = x + alpha * d; fx = new_fx
        if f_start - fx < tol * (abs(f_start) + tol):
            break
        new_dir = x - x_start                            # net direction of the sweep
        alpha = _line_min(f, x, new_dir)
        x = x + alpha * new_dir; fx = f(x)
        dirs[drop_idx] = dirs[-1]; dirs[-1] = new_dir    # rotate in the new direction
    return x


def _line_min(f, x, d, bracket=1.0):
    # golden-section line minimisation along direction d
    gr = (np.sqrt(5) - 1) / 2
    a, b = -bracket, bracket
    while abs(b - a) > 1e-6:
        c = b - gr * (b - a); e = a + gr * (b - a)
        if f(x + c * d) < f(x + e * d):
            b = e
        else:
            a = c
    return (a + b) / 2


def cross_entropy_method(f, x0, sigma0=1.0, pop_size=50, elite_frac=0.2,
                         max_iter=100, tol=1e-8, random_state=None):
    """Optimize by SAMPLING and refitting the elite (Rubinstein, 1997).

    A derivative-free, gradient-free global-ish method: keep a Gaussian over the
    search space, sample a population, keep the best ("elite") fraction, and REFIT
    the Gaussian to just those elites. The distribution marches toward the optimum
    and its variance shrinks as the elites agree. It is robust to noise and
    multimodality, needs no gradient, and is the workhorse behind many
    reinforcement-learning and planning methods (e.g. CEM-based MPC).
    """
    rng = np.random.RandomState(random_state) if not isinstance(
        random_state, np.random.RandomState) else random_state
    if rng is None:
        rng = np.random.RandomState()
    mean = np.asarray(x0, float).copy()
    sigma = np.full(len(mean), float(sigma0))
    n_elite = max(1, int(pop_size * elite_frac))
    for _ in range(max_iter):
        samples = rng.normal(mean, sigma, (pop_size, len(mean)))
        scores = np.array([f(s) for s in samples])
        elite = samples[np.argsort(scores)[:n_elite]]    # lowest f = best
        mean = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 1e-9                 # refit to the elite
        if np.max(sigma) < tol:
            break
    return mean


__all__ = ["lbfgs", "levenberg_marquardt", "frank_wolfe", "powell",
           "cross_entropy_method"]
