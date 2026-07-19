"""Bilateral filter: edge-preserving smoothing by range weighting."""
import numpy as np
import scipy.ndimage as ndi

from .vision import _gray


def bilateral_filter(image, spatial_sigma=2.0, range_sigma=0.1, radius=3):
    """Smooth noise but KEEP edges, by weighting on intensity too (Tomasi, 1998).

    A Gaussian blur averages a pixel with its neighbours by DISTANCE alone, so it
    smears edges. The bilateral filter adds a second weight on INTENSITY difference: a
    neighbour only contributes if it is both nearby AND similar in value. So within a
    smooth region it averages away noise, but across an edge the dissimilar side is
    down-weighted to nothing -- the edge survives. It is the classic edge-preserving
    smoother, and the intuition behind many later ones. ``spatial_sigma`` sets the
    neighbourhood, ``range_sigma`` how much intensity difference is tolerated.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    H, W = img.shape
    out = np.zeros_like(img)
    wsum = np.zeros_like(img)
    ax = np.arange(-radius, radius + 1)
    for dy in ax:
        for dx in ax:
            spatial = np.exp(-(dy ** 2 + dx ** 2) / (2 * spatial_sigma ** 2))
            shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            rng_w = np.exp(-((img - shifted) ** 2) / (2 * range_sigma ** 2))
            w = spatial * rng_w                          # spatial x range weight
            out += w * shifted; wsum += w
    return out / wsum


__all__ = ["bilateral_filter"]
