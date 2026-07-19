"""A contour that RELAXES onto an edge (Kass, Witkin & Terzopoulos, 1988)."""
import numpy as np
from .vision import harris_corners, corner_peaks, _gray


def active_contour(image, init_snake, alpha=0.1, beta=0.1, gamma=0.1,
                   w_edge=1.0, balloon=0.0, n_iter=200):
    """A contour that RELAXES onto an edge (Kass, Witkin & Terzopoulos, 1988).

    A "snake" is a closed curve pulled by two competing energies: an INTERNAL
    energy that keeps it smooth and short (first/second-derivative terms weighted by
    ``alpha`` and ``beta``) and an EXTERNAL energy that drags it toward strong image
    gradients. Minimising their sum, the curve settles onto the object boundary,
    interpolating across gaps where the edge is weak. An optional ``balloon`` force
    inflates (positive) or deflates (negative) the curve along its normal so it
    reliably travels to a distant boundary instead of stalling in a flat region.
    Solved implicitly with the pentadiagonal internal-force matrix. ``init_snake``
    is an (N, 2) array of (row, col) points enclosing the object.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    gy, gx = np.gradient(img)
    edge = gx ** 2 + gy ** 2                              # external potential
    fy, fx = np.gradient(edge)                            # pulls toward strong edges
    snake = np.array(init_snake, float)
    n = len(snake)
    # pentadiagonal matrix A for internal (smoothness) forces on a closed curve
    a = np.roll(np.eye(n), -1, 0) + np.roll(np.eye(n), 1, 0) - 2 * np.eye(n)
    b = (np.roll(np.eye(n), -2, 0) + np.roll(np.eye(n), 2, 0)
         - 4 * np.roll(np.eye(n), -1, 0) - 4 * np.roll(np.eye(n), 1, 0)
         + 6 * np.eye(n))
    A = -alpha * a + beta * b
    inv = np.linalg.inv(np.eye(n) + gamma * A)
    for _ in range(n_iter):
        r = np.clip(snake[:, 0].astype(int), 0, img.shape[0] - 1)
        c = np.clip(snake[:, 1].astype(int), 0, img.shape[1] - 1)
        force = np.column_stack([fy[r, c], fx[r, c]]) * w_edge
        if balloon:
            centroid = snake.mean(axis=0)                 # inward/outward normal
            normal = centroid - snake
            normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9
            force = force + balloon * normal              # +: deflate toward centre
        snake = inv @ (snake + gamma * force)
        snake[:, 0] = np.clip(snake[:, 0], 0, img.shape[0] - 1)
        snake[:, 1] = np.clip(snake[:, 1], 0, img.shape[1] - 1)
    return snake


__all__ = ["active_contour"]
