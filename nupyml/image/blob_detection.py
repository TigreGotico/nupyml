"""Find bright/dark regions across SCALE (Lindeberg, 1998)."""
import numpy as np
import scipy.ndimage as ndi
from .vision import _gray


def blob_detection(image, min_sigma=1.0, max_sigma=8.0, n_scales=10,
                   threshold=0.05):
    """Find bright/dark regions across SCALE (Lindeberg, 1998).

    A blob detector must work at whatever SIZE the blob happens to be. This builds a
    scale space of Laplacian-of-Gaussian responses -- the LoG is a blob-shaped filter
    whose strongest response occurs when its scale MATCHES the blob's size -- and
    finds local maxima jointly over position AND scale. Each maximum is a blob with a
    location and a radius read off its scale (``r = sigma * sqrt(2)``). It is the
    scale-selection principle behind SIFT's keypoints. Returns ``(row, col, radius)``
    per blob.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    sigmas = np.geomspace(min_sigma, max_sigma, n_scales)
    # scale-normalised LoG so responses are comparable across scales
    cube = np.stack([-s ** 2 * ndi.gaussian_laplace(img, s) for s in sigmas])
    blobs = []
    for si in range(n_scales):
        local = ndi.maximum_filter(cube, size=(3, 3, 3))
        peaks = (cube[si] == local[si]) & (cube[si] > threshold)
        for y, x in zip(*np.where(peaks)):
            blobs.append((y, x, sigmas[si] * np.sqrt(2)))
    return np.array(blobs)


__all__ = ["blob_detection"]
