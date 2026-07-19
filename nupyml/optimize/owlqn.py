"""L-BFGS extended to an L1 penalty (Andrew & Gao, 2007)."""
import numpy as np


def _orthant_project(x_old, x_new):
    # clamp any coordinate that crossed zero back to exactly zero
    crossed = np.sign(x_new) != np.sign(x_old)
    keep_zero = crossed & (x_old != 0)
    out = x_new.copy()
    out[keep_zero] = 0.0
    return out


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


__all__ = ["owlqn"]
