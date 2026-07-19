"""Rebuild the image from its Laplacian pyramid."""
import numpy as np
import scipy.ndimage as ndi


def _upsample(img, shape):
    out = np.zeros(shape)
    out[::2, ::2] = img[:(shape[0] + 1) // 2, :(shape[1] + 1) // 2]
    return ndi.gaussian_filter(out, 1.0) * 4             # interpolate the zeros


def pyramid_reconstruct(lap):
    """Rebuild the image from its Laplacian pyramid."""
    img = lap[-1]
    for level in reversed(lap[:-1]):
        img = level + _upsample(img, level.shape)
    return img


__all__ = ["pyramid_reconstruct"]
