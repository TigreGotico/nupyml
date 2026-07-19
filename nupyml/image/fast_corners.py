"""Detect corners with a cheap SEGMENT TEST (Rosten & Drummond, 2006)."""
import numpy as np
from .vision import _gray


def _has_run(mask, n):
    run = 0
    for v in mask:
        run = run + 1 if v else 0
        if run >= n:
            return True
    return False


_FAST_CIRCLE = [(0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
                (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2),
                (-1, 3)]


def fast_corners(image, threshold=0.15, n_contiguous=9):
    """Detect corners with a cheap SEGMENT TEST (Rosten & Drummond, 2006).

    Harris corners need image derivatives and eigenvalues per pixel. FAST asks a
    much cheaper question: on the ring of 16 pixels around a candidate, are there at
    least ``n_contiguous`` in a row that are ALL brighter (or ALL darker) than the
    centre by a threshold? If so, it is a corner. The contiguity requirement is what
    makes it fire on corners but not edges, and the test rejects most pixels after
    only a few comparisons -- fast enough for real-time tracking, which is why it is
    the 'F' in ORB. Returns corner (row, col) coordinates.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    H, W = img.shape
    corners = []
    ring = _FAST_CIRCLE + _FAST_CIRCLE[:n_contiguous]    # wrap for contiguity
    for y in range(3, H - 3):
        for x in range(3, W - 3):
            c = img[y, x]
            vals = np.array([img[y + dy, x + dx] for dy, dx in ring])
            brighter = vals > c + threshold
            darker = vals < c - threshold
            if _has_run(brighter, n_contiguous) or _has_run(darker, n_contiguous):
                corners.append((y, x))
    return np.array(corners)


__all__ = ["fast_corners"]
