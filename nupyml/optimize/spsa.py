"""Estimate a gradient from TWO function evaluations (Spall, 1992)."""
import numpy as np
from .. utils import check_random_state


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


__all__ = ["spsa"]
