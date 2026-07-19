"""The DETAIL lost at each pyramid level -- and it inverts exactly."""
import numpy as np
import scipy.ndimage as ndi
from .gaussian_pyramid import gaussian_pyramid
from .pyramid_reconstruct import pyramid_reconstruct


def _upsample(img, shape):
    out = np.zeros(shape)
    out[::2, ::2] = img[:(shape[0] + 1) // 2, :(shape[1] + 1) // 2]
    return ndi.gaussian_filter(out, 1.0) * 4             # interpolate the zeros


def laplacian_pyramid(image, levels=4, sigma=1.0):
    """The DETAIL lost at each pyramid level -- and it inverts exactly.

    Each Laplacian level is a Gaussian level minus the upsampled next-coarser level:
    the band of detail that downsampling threw away. Storing the small top of the
    Gaussian pyramid plus these detail bands lets you REBUILD the original exactly
    (``pyramid_reconstruct``), which is why it underlies image compression and
    seamless blending. Returns the detail bands plus the coarsest Gaussian level.
    """
    g = gaussian_pyramid(image, levels, sigma)
    lap = []
    for i in range(levels - 1):
        up = _upsample(g[i + 1], g[i].shape)
        lap.append(g[i] - up)                             # detail band
    lap.append(g[-1])                                     # coarsest residual
    return lap


__all__ = ["laplacian_pyramid"]
