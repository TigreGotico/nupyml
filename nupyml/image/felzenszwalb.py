"""Graph-based segmentation with an adaptive merge criterion (Felzenszwalb, 2004)."""
import numpy as np
import scipy.ndimage as ndi
from .vision import _gray


def felzenszwalb(image, scale=100.0, sigma=0.5, min_size=20):
    """Graph-based segmentation with an adaptive merge criterion (Felzenszwalb, 2004).

    Treat the image as a graph -- pixels are nodes, edges join neighbours with a
    weight equal to their intensity difference. Sort the edges and, cheapest first,
    MERGE the two components an edge joins UNLESS the edge is stronger than both
    components' current internal variation (plus a size-dependent slack ``scale``).
    This adapts the boundary threshold to each region's own texture, so it keeps
    high-variability regions whole while still splitting genuine edges -- a fast,
    parameter-light superpixel/segmentation method. Returns an integer label map.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    img = ndi.gaussian_filter(img, sigma)
    H, W = img.shape
    parent = np.arange(H * W)
    size = np.ones(H * W)
    intdiff = np.zeros(H * W)                            # internal difference

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    edges = []
    for y in range(H):
        for x in range(W):
            i = y * W + x
            if x + 1 < W:
                edges.append((abs(img[y, x] - img[y, x + 1]), i, i + 1))
            if y + 1 < H:
                edges.append((abs(img[y, x] - img[y + 1, x]), i, i + W))
    edges.sort()
    for w, a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        thresh_a = intdiff[ra] + scale / size[ra]
        thresh_b = intdiff[rb] + scale / size[rb]
        if w <= min(thresh_a, thresh_b):                 # merge if within tolerance
            parent[rb] = ra
            size[ra] += size[rb]
            intdiff[ra] = w
    # enforce a minimum component size
    for w, a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb and (size[ra] < min_size or size[rb] < min_size):
            parent[rb] = ra; size[ra] += size[rb]
    labels = np.array([find(i) for i in range(H * W)])
    _, labels = np.unique(labels, return_inverse=True)
    return labels.reshape(H, W)


# ------------------------------------------------------------------- homography


__all__ = ["felzenszwalb"]
