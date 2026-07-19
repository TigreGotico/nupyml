"""Minimise using only function VALUES -- no gradient (Powell, 1964)."""
import numpy as np


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


__all__ = ["powell"]
