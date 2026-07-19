"""A stack of the image at HALVING resolutions (Burt & Adelson, 1983)."""
import numpy as np
import scipy.ndimage as ndi


def gaussian_pyramid(image, levels=4, sigma=1.0):
    """A stack of the image at HALVING resolutions (Burt & Adelson, 1983).

    Blur then downsample, repeatedly. Each level halves the resolution, so features
    too big to see in one convolution at full scale become small enough at a coarser
    level -- the basis of multi-scale detection, blending, and coarse-to-fine
    search. Returns the list of images, finest first.
    """
    img = np.asarray(image, float)
    pyr = [img]
    for _ in range(levels - 1):
        img = ndi.gaussian_filter(img, sigma)[::2, ::2]   # blur then subsample
        pyr.append(img)
    return pyr


__all__ = ["gaussian_pyramid"]
