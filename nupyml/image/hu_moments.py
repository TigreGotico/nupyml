"""Hu moments: shape descriptors invariant to translation, rotation and scale."""
import numpy as np

from .vision import _gray


def hu_moments(image):
    """Seven shape descriptors invariant to translation, rotation, scale (Hu, 1962).

    To recognise a shape regardless of where it sits, how big it is, or which way it is
    turned, you need descriptors that do not change under those transforms. Hu built
    seven such invariants out of NORMALISED CENTRAL MOMENTS: centring removes
    translation, scale-normalising removes size, and specific polynomial combinations
    cancel rotation. They are a compact, classic signature for template matching and
    optical character recognition. Returns the 7 (log-scaled) Hu moments.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W]
    m00 = img.sum()
    xc = (x * img).sum() / m00; yc = (y * img).sum() / m00   # centroid
    xn, yn = x - xc, y - yc

    def mu(p, q):
        return (xn ** p * yn ** q * img).sum()

    def eta(p, q):
        return mu(p, q) / m00 ** (1 + (p + q) / 2)           # scale-normalised

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n03, n21, n12 = eta(3, 0), eta(0, 3), eta(2, 1), eta(1, 2)
    h = np.zeros(7)
    h[0] = n20 + n02
    h[1] = (n20 - n02) ** 2 + 4 * n11 ** 2
    h[2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h[4] = ((n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
            + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    h[5] = ((n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2)
            + 4 * n11 * (n30 + n12) * (n21 + n03))
    h[6] = ((3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
            - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    return -np.sign(h) * np.log10(np.abs(h) + 1e-30)         # log-scaled for range


__all__ = ["bilateral_filter", "non_local_means", "total_variation_denoise",
           "anisotropic_diffusion", "graph_cut_segmentation", "hu_moments"]


__all__ = ["hu_moments"]
