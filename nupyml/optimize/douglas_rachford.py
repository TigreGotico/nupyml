"""Douglas-Rachford splitting: minimise a sum of two non-smooth convex terms."""
import numpy as np


def douglas_rachford(prox_f, prox_g, x0, gamma=1.0, max_iter=500, tol=1e-9):
    """Minimise ``f(x) + g(x)`` using only each term's PROXIMAL operator (Lions & Mercier, 1979).

    Many problems are a sum of two convex pieces that are each simple ALONE but awkward
    together -- a data term plus an L1 penalty, a smooth loss plus an indicator of a
    constraint set. Douglas-Rachford never touches ``f + g`` directly; it alternates the
    two proximal operators (each a small, often closed-form, sub-problem) with a
    reflection-and-average update that provably converges to a minimiser of the sum, even
    when neither term is differentiable. It underlies ADMM and modern splitting methods.
    ``prox_f`` and ``prox_g`` are the proximal operators of the two terms.
    """
    y = np.array(x0, float)
    x = prox_f(y, gamma)
    for _ in range(max_iter):
        x = prox_f(y, gamma)
        z = prox_g(2 * x - y, gamma)                       # reflect through prox_f
        y_new = y + (z - x)                                # averaged update
        if np.linalg.norm(y_new - y) < tol:
            y = y_new
            break
        y = y_new
    return prox_f(y, gamma)


__all__ = ["douglas_rachford"]
