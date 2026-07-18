"""Vision v3: scale-invariant keypoints, dense flow, graph segmentation, geometry,
and cascade detection.

Five more vision algorithms. SIFT finds and describes keypoints that survive
SCALE and rotation. Horn-Schunck estimates a DENSE optical-flow field with a global
smoothness prior. Felzenszwalb segments by a graph criterion. Homography+RANSAC
fits the projective map between two views robustly to outliers. Viola-Jones detects
objects with a boosted cascade of Haar features over an integral image.
"""
import numpy as np
import scipy.ndimage as ndi

from ..base import BaseEstimator
from ..utils import check_random_state
from .vision import _gray
from . import integral_image, rectangle_sum


# --------------------------------------------------------------------------- SIFT
def sift(image, n_octaves=3, n_scales=3, sigma=1.6, contrast_threshold=0.03,
         n_keypoints=100):
    """Scale-invariant keypoints and descriptors (Lowe, 2004).

    A corner detector fires at one scale only, so a zoomed object is missed. SIFT
    searches a Gaussian SCALE-SPACE: it finds extrema of the difference-of-Gaussians
    across both position AND scale (so a feature is detected at whatever size it
    appears), assigns each a dominant ORIENTATION from local gradients, and encodes
    the surrounding gradients as a 128-D histogram descriptor -- normalised so it is
    invariant to scale, rotation, and illumination. Matching those descriptors is
    how images are stitched and objects recognised across viewpoints. Returns
    ``(keypoints, descriptors)``.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    keypoints = []
    for octave in range(n_octaves):
        scaled = img[::2 ** octave, ::2 ** octave]
        if min(scaled.shape) < 8:
            break
        gaussians = [ndi.gaussian_filter(scaled, sigma * 1.6 ** s)
                     for s in range(n_scales + 2)]
        dogs = [gaussians[s + 1] - gaussians[s] for s in range(n_scales + 1)]
        for s in range(1, n_scales):
            prev, cur, nxt = dogs[s - 1], dogs[s], dogs[s + 1]
            H, W = cur.shape
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    v = cur[y, x]
                    if abs(v) < contrast_threshold:
                        continue
                    patch = np.stack([prev[y - 1:y + 2, x - 1:x + 2],
                                      cur[y - 1:y + 2, x - 1:x + 2],
                                      nxt[y - 1:y + 2, x - 1:x + 2]])
                    if v >= patch.max() or v <= patch.min():   # 26-neighbour extremum
                        keypoints.append((y * 2 ** octave, x * 2 ** octave,
                                          sigma * 1.6 ** s * 2 ** octave))
    keypoints = keypoints[:n_keypoints]
    desc = np.array([_sift_descriptor(img, y, x, sc) for (y, x, sc) in keypoints]) \
        if keypoints else np.empty((0, 128))
    return np.array(keypoints), desc


def _sift_descriptor(img, y, x, scale, size=16):
    H, W = img.shape
    half = size // 2
    y0, x0 = int(np.clip(y, half, H - half - 1)), int(np.clip(x, half, W - half - 1))
    win = img[y0 - half:y0 + half, x0 - half:x0 + half]
    gy, gx = np.gradient(win)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) + np.pi)                  # [0, 2pi)
    dominant = ang.ravel()[mag.argmax()]               # orientation normalisation
    ang = (ang - dominant) % (2 * np.pi)
    desc = []
    cell = size // 4
    for i in range(4):
        for j in range(4):
            m = mag[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            a = ang[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            hist, _ = np.histogram(a, bins=8, range=(0, 2 * np.pi), weights=m)
            desc.extend(hist)
    desc = np.array(desc)
    return desc / (np.linalg.norm(desc) + 1e-8)


# ------------------------------------------------------------------ optical flow
def horn_schunck(frame1, frame2, alpha=1.0, n_iter=100):
    """DENSE optical flow with a global smoothness prior (Horn & Schunck, 1981).

    Lucas-Kanade estimates flow only at good corners. Horn-Schunck instead solves
    for a flow vector at EVERY pixel by minimising, globally, the brightness-
    constancy error plus a SMOOTHNESS penalty (neighbouring flow vectors should
    agree). ``alpha`` weights the smoothness: larger gives a smoother, more filled-in
    field, which is what propagates motion into the textureless regions Lucas-Kanade
    leaves blank. Solved by the classic Jacobi iteration. Returns the ``(u, v)`` flow
    fields.
    """
    f1 = np.asarray(frame1, float); f2 = np.asarray(frame2, float)
    Ix = ndi.sobel(f1, axis=1) / 8.0
    Iy = ndi.sobel(f1, axis=0) / 8.0
    It = f2 - f1
    u = np.zeros_like(f1); v = np.zeros_like(f1)
    kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                       [1 / 6, 0, 1 / 6],
                       [1 / 12, 1 / 6, 1 / 12]])
    for _ in range(n_iter):
        u_bar = ndi.convolve(u, kernel, mode="nearest")
        v_bar = ndi.convolve(v, kernel, mode="nearest")
        denom = alpha ** 2 + Ix ** 2 + Iy ** 2
        common = (Ix * u_bar + Iy * v_bar + It) / denom
        u = u_bar - Ix * common                          # Jacobi update
        v = v_bar - Iy * common
    return u, v


# --------------------------------------------------------------- felzenszwalb
def felzenszwalb(image, scale=100.0, sigma=0.5, min_size=20):
    """Graph-based segmentation with an adaptive merge criterion (Felzenszwalb, 2004).

    Treat the image as a graph -- pixels are nodes, edges join neighbours with a
    weight equal to their intensity difference. Sort the edges and, cheapest first,
    MERGE the two components an edge joins UNLESS the edge is stronger than both
    components' current internal variation (plus a size-dependent slack ``scale``).
    This adapts the boundary threshold to each region's own texture, so it keeps
    high-variability regions whole while still splitting genuine edges -- a fast,
    parameter-light superpixel/segmentation method. Returns an integer label map.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    img = ndi.gaussian_filter(img, sigma)
    H, W = img.shape
    parent = np.arange(H * W)
    size = np.ones(H * W)
    intdiff = np.zeros(H * W)                            # internal difference

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    edges = []
    for y in range(H):
        for x in range(W):
            i = y * W + x
            if x + 1 < W:
                edges.append((abs(img[y, x] - img[y, x + 1]), i, i + 1))
            if y + 1 < H:
                edges.append((abs(img[y, x] - img[y + 1, x]), i, i + W))
    edges.sort()
    for w, a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        thresh_a = intdiff[ra] + scale / size[ra]
        thresh_b = intdiff[rb] + scale / size[rb]
        if w <= min(thresh_a, thresh_b):                 # merge if within tolerance
            parent[rb] = ra
            size[ra] += size[rb]
            intdiff[ra] = w
    # enforce a minimum component size
    for w, a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb and (size[ra] < min_size or size[rb] < min_size):
            parent[rb] = ra; size[ra] += size[rb]
    labels = np.array([find(i) for i in range(H * W)])
    _, labels = np.unique(labels, return_inverse=True)
    return labels.reshape(H, W)


# ------------------------------------------------------------------- homography
def estimate_homography(src, dst):
    """The 3x3 projective map from >=4 point correspondences, via the DLT."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)                              # null space of A
    return H / H[2, 2]


def ransac_homography(src, dst, threshold=3.0, n_iter=1000, random_state=None):
    """Fit a homography ROBUSTLY, rejecting outlier matches (Fischler & Bolles, 1981).

    Feature matches always contain wrong pairs, and least-squares lets a single
    outlier ruin the fit. RANSAC instead samples minimal sets (4 correspondences),
    fits a candidate homography to each, and keeps the one with the most INLIERS
    (matches it explains within ``threshold`` pixels) -- then refits on all inliers.
    Because it needs only a handful of correct matches in the sample, it tolerates a
    large fraction of outliers. Returns ``(H, inlier_mask)``.
    """
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    rng = check_random_state(random_state)
    n = len(src)
    best_H, best_inliers = None, np.zeros(n, bool)

    def apply(H, pts):
        p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
        return p[:, :2] / p[:, 2:3]

    for _ in range(n_iter):
        idx = rng.choice(n, 4, replace=False)
        try:
            H = estimate_homography(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        err = np.linalg.norm(apply(H, src) - dst, axis=1)
        inliers = err < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers, best_H = inliers, H
    if best_inliers.sum() >= 4:                          # refit on all inliers
        best_H = estimate_homography(src[best_inliers], dst[best_inliers])
    return best_H, best_inliers


# ------------------------------------------------------------------ Viola-Jones
def _haar_features(ii, window, step=4):
    # two-rectangle horizontal/vertical Haar features at several sizes and
    # positions across the window (so features cover corners AND the centre)
    feats = []
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    for fs in (window // 4, window // 2):
        if fs < 2:
            continue
        h = fs // 2
        for y in range(0, H - fs + 1, step):
            for x in range(0, W - fs + 1, step):
                left = rectangle_sum(ii, y, x, y + fs, x + h)
                right = rectangle_sum(ii, y, x + h, y + fs, x + fs)
                feats.append(right - left)
                top = rectangle_sum(ii, y, x, y + h, x + fs)
                bot = rectangle_sum(ii, y + h, x, y + fs, x + fs)
                feats.append(bot - top)
    return np.array(feats)


class ViolaJones(BaseEstimator):
    """Detect objects with a boosted cascade of Haar features (Viola & Jones, 2001).

    The breakthrough that made real-time face detection possible. Features are
    simple HAAR rectangles (sums of pixel intensities in adjacent boxes), each
    computable in constant time from the INTEGRAL IMAGE. AdaBoost picks the few
    features that best separate object from background and weights them; arranging
    the resulting stumps as a CASCADE lets the easy negatives be rejected in the
    first cheap stages. Here: Haar features over the integral image + an AdaBoost
    stump classifier trained to detect a pattern (e.g. a bright central region).
    """

    def __init__(self, window=24, n_rounds=30, step=4):
        self.window = window
        self.n_rounds = n_rounds
        self.step = step

    def _features(self, images):
        return np.array([_haar_features(integral_image(im), self.window, self.step)
                         for im in images])

    def fit(self, images, y):
        X = self._features(images)
        y = np.where(np.asarray(y) > 0, 1.0, -1.0)
        n, m = X.shape
        w = np.ones(n) / n
        self.stumps_ = []
        for _ in range(self.n_rounds):
            best = None
            for j in range(m):
                thr = np.median(X[:, j])
                for polarity in (1, -1):
                    pred = np.where(polarity * (X[:, j] - thr) > 0, 1.0, -1.0)
                    err = w[pred != y].sum()
                    if best is None or err < best[0]:
                        best = (err, j, thr, polarity)
            err, j, thr, pol = best
            err = min(max(err, 1e-6), 1 - 1e-6)
            alpha = 0.5 * np.log((1 - err) / err)
            self.stumps_.append((j, thr, pol, alpha))
            pred = np.where(pol * (X[:, j] - thr) > 0, 1.0, -1.0)
            w = w * np.exp(-alpha * y * pred)
            w /= w.sum()
        return self

    def decision_function(self, images):
        X = self._features(images)
        score = np.zeros(len(X))
        for j, thr, pol, alpha in self.stumps_:
            score += alpha * np.where(pol * (X[:, j] - thr) > 0, 1.0, -1.0)
        return score

    def predict(self, images):
        return (self.decision_function(images) > 0).astype(int)


__all__ = ["sift", "horn_schunck", "felzenszwalb", "estimate_homography",
           "ransac_homography", "ViolaJones"]
