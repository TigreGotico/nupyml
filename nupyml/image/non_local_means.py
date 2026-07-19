"""Non-local means: denoise by averaging similar patches from across the image."""
import numpy as np
import scipy.ndimage as ndi

from .vision import _gray


def non_local_means(image, patch_radius=2, search_radius=5, h=0.1):
    """Denoise by averaging SIMILAR PATCHES from across the image (Buades, 2005).

    Most denoisers use only nearby pixels; non-local means uses SELF-SIMILARITY. To
    restore a pixel it compares the small PATCH around it to patches all over the image
    and averages their centres, weighted by patch similarity -- so the many repetitions
    of texture and structure in a natural image reinforce each other and cancel the
    (independent) noise. It set the standard for denoising quality before deep methods.
    ``h`` controls how quickly the weight falls off with patch distance.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    H, W = img.shape
    pad = patch_radius
    padded = np.pad(img, pad, mode="reflect")
    out = np.zeros_like(img); wsum = np.zeros_like(img)

    def patch(y, x):
        return padded[y:y + 2 * pad + 1, x:x + 2 * pad + 1]

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            sp = np.pad(shifted, pad, mode="reflect")
            # squared patch distance via a uniform box filter of the pixel diff^2
            diff2 = (padded - sp) ** 2
            pd = ndi.uniform_filter(diff2, 2 * pad + 1)[pad:-pad, pad:-pad]
            w = np.exp(-pd / (h ** 2))
            out += w * shifted; wsum += w
    return out / wsum


__all__ = ["non_local_means"]
