"""Vote for CIRCLES in an accumulator (Duda & Hart, 1972)."""
import numpy as np
import scipy.ndimage as ndi
from .vision import _gray


def hough_circles(image, radii, threshold=0.4, edge_percentile=90):
    """Vote for CIRCLES in an accumulator (Duda & Hart, 1972).

    A circle of known radius ``r`` centred at ``(a, b)`` means every edge point on it
    is exactly ``r`` from ``(a, b)``. So each edge pixel VOTES for all the centres it
    could belong to -- a circle of radius ``r`` around itself -- and true centres
    accumulate many votes where many edge pixels agree. Sweeping ``r`` finds circles
    of each size. Robust to occlusion and noise because it is a consensus over
    independent votes. Returns detected ``(row, col, radius)`` circles.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    gy, gx = np.gradient(img)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edges = mag > np.percentile(mag, edge_percentile)
    ys, xs = np.where(edges)
    H, W = img.shape
    circles = []
    thetas = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    ct, st = np.cos(thetas), np.sin(thetas)
    for r in radii:
        acc = np.zeros((H, W))
        for y, x in zip(ys, xs):
            a = np.round(y - r * st).astype(int)         # candidate centres
            b = np.round(x - r * ct).astype(int)
            ok = (a >= 0) & (a < H) & (b >= 0) & (b < W)
            np.add.at(acc, (a[ok], b[ok]), 1)
        acc /= len(thetas)
        peak = ndi.maximum_filter(acc, size=5)
        for y, x in zip(*np.where((acc == peak) & (acc > threshold))):
            circles.append((y, x, r))
    return np.array(circles)


__all__ = ["hough_circles"]
