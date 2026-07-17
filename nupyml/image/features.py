"""Classic image feature descriptors: HOG, LBP, GLCM/Haralick, Gabor.

These are the hand-designed descriptors that dominated computer vision before
deep nets -- and they still matter: they are interpretable, need no training,
and work on tiny datasets where a CNN would overfit. Each turns a raw pixel
grid (a 2-D float array) into a vector that is stable under the nuisance the
descriptor is designed to ignore.
"""
import numpy as np


def _as_gray(image):
    image = np.asarray(image, dtype=float)
    if image.ndim == 3:                       # collapse colour to luminance
        image = image.mean(axis=2)
    return image


def histogram_of_oriented_gradients(image, orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2)):
    """HOG: describe local shape by the DISTRIBUTION of gradient directions.

    THE IDEA
    --------
    Object appearance is captured well by the directions of intensity edges --
    a pedestrian is a particular arrangement of near-vertical limb edges. HOG
    computes the gradient at every pixel, bins the gradient DIRECTIONS (weighted
    by magnitude) into a histogram over small cells, then normalises those
    histograms over larger overlapping blocks.

    WHY IT WORKS
    ------------
    Binning directions inside a cell makes the descriptor tolerant to small
    shifts and deformations (the exact edge location within the cell does not
    matter, only that an edge of that orientation is present). Block
    normalisation removes the effect of illumination and contrast -- what
    survives is pure local SHAPE. This is the descriptor behind the classic
    Dalal-Triggs pedestrian detector.

    Returns a 1-D feature vector.
    """
    image = _as_gray(image)
    # gradients (simple centred differences)
    gy, gx = np.gradient(image)
    magnitude = np.hypot(gx, gy)
    # orientation in [0, 180) -- unsigned, since an edge and its reverse are the
    # same edge for shape purposes
    orientation = (np.rad2deg(np.arctan2(gy, gx)) % 180)

    cy, cx = pixels_per_cell
    n_cells_y = image.shape[0] // cy
    n_cells_x = image.shape[1] // cx
    cell_hist = np.zeros((n_cells_y, n_cells_x, orientations))
    bin_width = 180.0 / orientations
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            mag = magnitude[i * cy:(i + 1) * cy, j * cx:(j + 1) * cx]
            ori = orientation[i * cy:(i + 1) * cy, j * cx:(j + 1) * cx]
            b = np.minimum((ori // bin_width).astype(int), orientations - 1)
            # magnitude-weighted vote into the orientation bin
            for k in range(orientations):
                cell_hist[i, j, k] = mag[b == k].sum()

    by, bx = cells_per_block
    blocks = []
    for i in range(n_cells_y - by + 1):
        for j in range(n_cells_x - bx + 1):
            block = cell_hist[i:i + by, j:j + bx, :].ravel()
            # L2 block normalisation -> contrast/illumination invariance
            block = block / np.sqrt((block ** 2).sum() + 1e-10)
            blocks.append(block)
    return np.concatenate(blocks) if blocks else cell_hist.ravel()


def local_binary_pattern(image, n_points=8, radius=1):
    """LBP: describe texture by comparing each pixel to its ring of neighbours.

    THE IDEA
    --------
    For every pixel, look at ``n_points`` neighbours on a circle of the given
    radius. Each neighbour brighter-than-or-equal-to the centre contributes a 1,
    darker a 0; reading those bits around the circle gives an integer CODE per
    pixel. The image's texture is then the HISTOGRAM of those codes.

    WHY IT WORKS
    ------------
    The comparison is to the CENTRE pixel, so the code is invariant to any
    monotonic change in illumination (brighten the whole patch and every
    comparison is unchanged). What it captures is the local micro-pattern -- edge,
    spot, flat, corner -- which is exactly what "texture" means. Cheap, powerful,
    and the basis of many face-recognition and texture-classification pipelines.

    Returns the LBP code image (same shape as input, interior valid).
    """
    image = _as_gray(image)
    rows, cols = image.shape
    codes = np.zeros_like(image)
    # sample points evenly around the circle
    angles = 2 * np.pi * np.arange(n_points) / n_points
    dy = -radius * np.sin(angles)
    dx = radius * np.cos(angles)
    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            centre = image[i, j]
            code = 0
            for k in range(n_points):
                # nearest-neighbour sample of the circular point
                ni = int(round(i + dy[k]))
                nj = int(round(j + dx[k]))
                code |= (image[ni, nj] >= centre) << k
            codes[i, j] = code
    return codes


def graycomatrix(image, distance=1, angle=0.0, levels=8):
    """Gray-Level Co-occurrence Matrix: how often gray-level PAIRS co-occur.

    Quantise the image to ``levels`` gray levels, then count how often a pixel of
    level ``i`` sits at the given (distance, angle) offset from a pixel of level
    ``j``. The resulting ``levels x levels`` matrix is the joint distribution of
    neighbouring intensities -- the raw material Haralick features summarise. A
    smooth texture concentrates mass on the diagonal (neighbours are similar); a
    busy one spreads it off-diagonal.
    """
    image = _as_gray(image)
    # quantise to a small number of levels so the matrix is dense enough to count
    lo, hi = image.min(), image.max()
    q = np.floor((image - lo) / (hi - lo + 1e-12) * (levels - 1)).astype(int)
    q = np.clip(q, 0, levels - 1)
    di = int(round(-distance * np.sin(angle)))
    dj = int(round(distance * np.cos(angle)))
    glcm = np.zeros((levels, levels))
    rows, cols = q.shape
    for i in range(rows):
        for j in range(cols):
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                glcm[q[i, j], q[ni, nj]] += 1
    return glcm


def haralick_features(glcm):
    """Summarise a GLCM into interpretable texture scalars (Haralick, 1973).

    A full co-occurrence matrix is too big to use directly; Haralick's insight
    was that a handful of scalar summaries capture the texture: contrast (how
    different neighbours are), homogeneity (mass near the diagonal), energy (how
    ordered/repetitive), correlation (linear predictability of a neighbour), and
    entropy (randomness). Returns a dict of these.
    """
    p = glcm / (glcm.sum() + 1e-12)          # normalise to a probability
    levels = p.shape[0]
    i = np.arange(levels)[:, None]
    j = np.arange(levels)[None, :]
    mu_i = (i * p).sum()
    mu_j = (j * p).sum()
    si = np.sqrt(((i - mu_i) ** 2 * p).sum())
    sj = np.sqrt(((j - mu_j) ** 2 * p).sum())
    return {
        "contrast": ((i - j) ** 2 * p).sum(),
        "homogeneity": (p / (1.0 + (i - j) ** 2)).sum(),
        "energy": (p ** 2).sum(),
        "correlation": (((i - mu_i) * (j - mu_j) * p).sum() / (si * sj + 1e-12)),
        "entropy": -(p * np.log(p + 1e-12)).sum(),
    }


def gabor_kernel(frequency, theta=0.0, sigma=None, size=None):
    """A Gabor filter: a sinusoid windowed by a Gaussian -- an oriented edge/bar
    detector at a chosen scale.

    Gabor filters model the receptive fields of early visual cortex: each
    responds strongly to intensity variation at a particular ORIENTATION and
    spatial FREQUENCY, and nowhere else. A bank of them at several orientations
    and scales gives a rich, biologically-motivated texture description. Returns
    the (real-part) kernel; convolve an image with it to get that channel's
    response.
    """
    if sigma is None:
        sigma = 1.0 / frequency
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    # rotate coordinates into the filter's orientation
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2))
    carrier = np.cos(2 * np.pi * frequency * xr)
    kernel = envelope * carrier
    return kernel - kernel.mean()            # zero-mean -> no DC response


def gabor_features(image, frequencies=(0.1, 0.2, 0.4), n_orientations=4):
    """Filter an image with a Gabor bank; return the (mean, std) of each response.

    A texture is described by how strongly it excites filters of each
    orientation and scale -- the summary statistics of the filtered images form a
    compact, rotation-aware texture vector.
    """
    from scipy.signal import fftconvolve
    image = _as_gray(image)
    feats = []
    for f in frequencies:
        for o in range(n_orientations):
            theta = np.pi * o / n_orientations
            k = gabor_kernel(f, theta)
            resp = fftconvolve(image, k, mode="same")
            feats.extend([resp.mean(), resp.std()])
    return np.array(feats)


__all__ = ["histogram_of_oriented_gradients", "local_binary_pattern",
           "graycomatrix", "haralick_features", "gabor_kernel", "gabor_features"]
