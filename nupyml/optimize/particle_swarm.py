"""Particle swarm: a population that steers toward good regions collectively."""
import numpy as np


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


__all__ = ["particle_swarm"]
