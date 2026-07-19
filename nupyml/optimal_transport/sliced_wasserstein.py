"""Average 1-D Wasserstein distance over many random PROJECTIONS."""
import numpy as np
from ..utils import check_array, check_random_state


def sliced_wasserstein(X, Y, n_projections=50, p=2, random_state=None):
    """Average 1-D Wasserstein distance over many random PROJECTIONS.

    THE TRICK
    ---------
    The Wasserstein distance in 1-D is trivial -- sort both samples and match them
    in order (no linear program). Sliced Wasserstein exploits that: project both
    point clouds onto a random direction, compute the cheap 1-D distance, and
    average over many directions. The result is a true metric that approximates
    the full Wasserstein distance while scaling to high dimensions and large
    samples, because it never builds an ``n*m`` cost matrix. The standard
    OT distance when you have many points in many dimensions.
    """
    X = check_array(X); Y = check_array(Y)
    rng = check_random_state(random_state)
    d = X.shape[1]
    total = 0.0
    for _ in range(n_projections):
        theta = rng.randn(d); theta /= np.linalg.norm(theta) + 1e-12
        xp = np.sort(X @ theta); yp = np.sort(Y @ theta)
        # match equal quantiles by resampling to a common length
        m = max(len(xp), len(yp))
        qx = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(xp)), xp)
        qy = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(yp)), yp)
        total += np.mean(np.abs(qx - qy) ** p)
    return float((total / n_projections) ** (1.0 / p))


__all__ = ["sliced_wasserstein"]
