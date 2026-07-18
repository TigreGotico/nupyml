"""Vision v4: fast corners, scale-space blobs, circle detection, two-view geometry,
and mean-shift segmentation.

Five more vision algorithms. FAST detects corners with a cheap segment test. Blob
detection finds bright/dark regions across SCALE. Hough circles vote for circular
shapes. The fundamental matrix relates two views of a scene (epipolar geometry).
Mean-shift segments an image by finding modes in colour+position space.
"""
import numpy as np
import scipy.ndimage as ndi

from ..base import BaseEstimator
from ..utils import check_random_state
from .vision import _gray


# Bresenham circle of radius 3 (the 16 pixels FAST tests), clockwise
_FAST_CIRCLE = [(0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
                (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2),
                (-1, 3)]


def fast_corners(image, threshold=0.15, n_contiguous=9):
    """Detect corners with a cheap SEGMENT TEST (Rosten & Drummond, 2006).

    Harris corners need image derivatives and eigenvalues per pixel. FAST asks a
    much cheaper question: on the ring of 16 pixels around a candidate, are there at
    least ``n_contiguous`` in a row that are ALL brighter (or ALL darker) than the
    centre by a threshold? If so, it is a corner. The contiguity requirement is what
    makes it fire on corners but not edges, and the test rejects most pixels after
    only a few comparisons -- fast enough for real-time tracking, which is why it is
    the 'F' in ORB. Returns corner (row, col) coordinates.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    H, W = img.shape
    corners = []
    ring = _FAST_CIRCLE + _FAST_CIRCLE[:n_contiguous]    # wrap for contiguity
    for y in range(3, H - 3):
        for x in range(3, W - 3):
            c = img[y, x]
            vals = np.array([img[y + dy, x + dx] for dy, dx in ring])
            brighter = vals > c + threshold
            darker = vals < c - threshold
            if _has_run(brighter, n_contiguous) or _has_run(darker, n_contiguous):
                corners.append((y, x))
    return np.array(corners)


def _has_run(mask, n):
    run = 0
    for v in mask:
        run = run + 1 if v else 0
        if run >= n:
            return True
    return False


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


def hough_circles(image, radii, threshold=0.4, edge_percentile=90):
    """Vote for CIRCLES in an accumulator (Duda & Hart, 1972).

    A circle of known radius ``r`` centred at ``(a, b)`` means every edge point on it
    is exactly ``r`` from ``(a, b)``. So each edge pixel VOTES for all the centres it
    could belong to -- a circle of radius ``r`` around itself -- and true centres
    accumulate many votes where many edge pixels agree. Sweeping ``r`` finds circles
    of each size. Robust to occlusion and noise because it is a consensus over
    independent votes. Returns detected ``(row, col, radius)`` circles.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    gy, gx = np.gradient(img)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edges = mag > np.percentile(mag, edge_percentile)
    ys, xs = np.where(edges)
    H, W = img.shape
    circles = []
    thetas = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    ct, st = np.cos(thetas), np.sin(thetas)
    for r in radii:
        acc = np.zeros((H, W))
        for y, x in zip(ys, xs):
            a = np.round(y - r * st).astype(int)         # candidate centres
            b = np.round(x - r * ct).astype(int)
            ok = (a >= 0) & (a < H) & (b >= 0) & (b < W)
            np.add.at(acc, (a[ok], b[ok]), 1)
        acc /= len(thetas)
        peak = ndi.maximum_filter(acc, size=5)
        for y, x in zip(*np.where((acc == peak) & (acc > threshold))):
            circles.append((y, x, r))
    return np.array(circles)


def _normalise_points(pts):
    mean = pts.mean(axis=0)
    scale = np.sqrt(2) / (np.linalg.norm(pts - mean, axis=1).mean() + 1e-12)
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]], [0, 0, 1]])
    ph = np.column_stack([pts, np.ones(len(pts))]) @ T.T
    return ph[:, :2], T


def fundamental_matrix(pts1, pts2):
    """The epipolar relation between two views: the normalised 8-POINT algorithm
    (Longuet-Higgins; Hartley).

    Two cameras see the same scene. The fundamental matrix ``F`` encodes their
    relative geometry: for every pair of corresponding image points,
    ``x2ᵀ F x1 = 0`` -- a point in one image must lie on a LINE (its epipolar line)
    in the other. Given >= 8 correspondences, ``F`` is the null space of a linear
    system; NORMALISING the coordinates first (Hartley) is essential for numerical
    stability, and ``F`` is forced to rank 2 afterwards. It is the foundation of
    stereo, structure-from-motion and 3-D reconstruction.
    """
    pts1 = np.asarray(pts1, float); pts2 = np.asarray(pts2, float)
    n1, T1 = _normalise_points(pts1)
    n2, T2 = _normalise_points(pts2)
    A = np.array([[x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
                  for (x1, y1), (x2, y2) in zip(n1, n2)])
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    U, S, Vt2 = np.linalg.svd(F)                          # enforce rank 2
    S[2] = 0
    F = U @ np.diag(S) @ Vt2
    F = T2.T @ F @ T1                                     # denormalise
    return F / (F[2, 2] if abs(F[2, 2]) > 1e-12 else np.linalg.norm(F))


def ransac_fundamental(pts1, pts2, threshold=0.02, n_iter=500, random_state=None):
    """Fit the fundamental matrix robustly to outlier correspondences."""
    pts1 = np.asarray(pts1, float); pts2 = np.asarray(pts2, float)
    rng = check_random_state(random_state)
    n = len(pts1)
    h1 = np.column_stack([pts1, np.ones(n)])
    h2 = np.column_stack([pts2, np.ones(n)])
    best_F, best_inliers = None, np.zeros(n, bool)
    for _ in range(n_iter):
        idx = rng.choice(n, 8, replace=False)
        try:
            F = fundamental_matrix(pts1[idx], pts2[idx])
        except np.linalg.LinAlgError:
            continue
        err = np.abs(np.sum(h2 * (h1 @ F.T), axis=1))    # |x2^T F x1|
        inliers = err < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers, best_F = inliers, F
    if best_inliers.sum() >= 8:
        best_F = fundamental_matrix(pts1[best_inliers], pts2[best_inliers])
    return best_F, best_inliers


class MeanShiftSegmentation(BaseEstimator):
    """Segment by finding MODES in colour+position space (Comaniciu & Meer, 2002).

    K-means needs the number of segments; mean-shift discovers it. Each pixel becomes
    a point in a joint COLOUR-and-POSITION space, and mean-shift climbs the density
    of those points -- each pixel iteratively moves to the weighted mean of its
    neighbours within a bandwidth -- until it reaches a MODE. Pixels that converge to
    the same mode form a segment, so the number of regions falls out of the data. The
    ``spatial_radius`` and ``color_radius`` bandwidths trade region size against
    detail. Returns an integer label map.
    """

    def __init__(self, spatial_radius=2.0, color_radius=0.3, max_iter=20):
        self.spatial_radius = spatial_radius
        self.color_radius = color_radius
        self.max_iter = max_iter

    def fit_predict(self, image):
        img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
        H, W = img.shape
        rows, cols = np.mgrid[0:H, 0:W]
        feats = np.column_stack([rows.ravel() / self.spatial_radius,
                                 cols.ravel() / self.spatial_radius,
                                 img.ravel() / self.color_radius])
        pts = feats.copy()
        for _ in range(self.max_iter):
            new = np.empty_like(pts)
            for i in range(len(pts)):
                d = np.linalg.norm(feats - pts[i], axis=1)
                near = d < 1.0
                new[i] = feats[near].mean(axis=0)        # mean shift toward the mode
            if np.abs(new - pts).max() < 1e-3:
                pts = new; break
            pts = new
        # group pixels whose modes coincide
        rounded = np.round(pts, 1)
        _, labels = np.unique(rounded, axis=0, return_inverse=True)
        return labels.reshape(H, W)


__all__ = ["fast_corners", "blob_detection", "hough_circles", "fundamental_matrix",
           "ransac_fundamental", "MeanShiftSegmentation"]
