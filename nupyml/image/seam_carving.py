"""Resize by removing the least important PATHS of pixels (Avidan, 2007)."""
import numpy as np
from .vision import harris_corners, corner_peaks, _gray


def seam_carving(image, n_seams, axis=1):
    """Resize by removing the least important PATHS of pixels (Avidan, 2007).

    Scaling squashes everything equally; cropping loses the edges. Seam carving
    removes, one at a time, the connected path of pixels ("seam") with the least
    ENERGY (gradient magnitude), so it shrinks an image through its bland regions
    and leaves the salient objects intact. Each seam is the minimum-cost top-to-
    bottom path found by dynamic programming. Set ``axis=0`` to carve horizontal
    seams (reduce height). Returns the carved image.
    """
    img = np.asarray(image, float)
    if axis == 0:
        return seam_carving(img.T if img.ndim == 2 else img.transpose(1, 0, 2),
                            n_seams, axis=1).T if img.ndim == 2 else \
            seam_carving(img.transpose(1, 0, 2), n_seams, axis=1).transpose(1, 0, 2)
    for _ in range(n_seams):
        gray = _gray(img) if img.ndim == 3 else img
        gy, gx = np.gradient(gray)
        energy = np.abs(gx) + np.abs(gy)
        H, W = energy.shape
        cost = energy.copy()
        back = np.zeros((H, W), int)
        for y in range(1, H):                            # DP for the min seam
            for x in range(W):
                lo, hi = max(0, x - 1), min(W, x + 2)
                k = lo + int(np.argmin(cost[y - 1, lo:hi]))
                back[y, x] = k
                cost[y, x] += cost[y - 1, k]
        seam = np.empty(H, int)
        seam[-1] = int(np.argmin(cost[-1]))
        for y in range(H - 2, -1, -1):
            seam[y] = back[y + 1, seam[y + 1]]
        # remove the seam column-by-row
        keep = np.ones((H, W), bool)
        keep[np.arange(H), seam] = False
        if img.ndim == 3:
            img = np.stack([img[:, :, c][keep].reshape(H, W - 1)
                            for c in range(img.shape[2])], axis=2)
        else:
            img = img[keep].reshape(H, W - 1)
    return img


__all__ = ["seam_carving"]
