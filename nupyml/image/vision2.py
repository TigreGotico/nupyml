"""Vision v2: pyramids, segmentation, contours, binary descriptors, visual words,
and content-aware resizing.

Six classic vision algorithms beyond the feature extractors already here. Pyramids
give scale. Watershed and active contours SEGMENT. ORB gives rotation-aware binary
keypoints for matching. Bag of visual words turns descriptors into a fixed vector
for classification. Seam carving resizes an image by removing its least important
pixels.
"""
import numpy as np
import scipy.ndimage as ndi

from ..base import BaseEstimator
from ..utils import check_random_state
from .vision import harris_corners, corner_peaks, _gray


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


def watershed(image, markers, mask=None):
    """Flood the intensity landscape from seed MARKERS until basins meet
    (Meyer, 1994).

    Read the image as a topography -- bright = high. Start water rising from each
    labelled marker; as the level rises, each pixel joins the basin whose water
    reaches it first, and basins are kept from merging at the ridges between them.
    Those ridges become the segment boundaries. It is the classic way to split
    touching objects (cells, coins) once you can seed each one. Priority-flood
    implementation over the marker labels.
    """
    import heapq
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    markers = np.asarray(markers, int)
    labels = markers.copy()
    H, W = img.shape
    heap = []
    for y, x in zip(*np.where(markers > 0)):
        heapq.heappush(heap, (img[y, x], int(markers[y, x]), y, x))
    while heap:
        _, lab, y, x = heapq.heappop(heap)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and labels[ny, nx] == 0:
                if mask is not None and not mask[ny, nx]:
                    continue
                labels[ny, nx] = lab                     # join this basin
                heapq.heappush(heap, (img[ny, nx], lab, ny, nx))
    return labels


def active_contour(image, init_snake, alpha=0.1, beta=0.1, gamma=0.1,
                   w_edge=1.0, balloon=0.0, n_iter=200):
    """A contour that RELAXES onto an edge (Kass, Witkin & Terzopoulos, 1988).

    A "snake" is a closed curve pulled by two competing energies: an INTERNAL
    energy that keeps it smooth and short (first/second-derivative terms weighted by
    ``alpha`` and ``beta``) and an EXTERNAL energy that drags it toward strong image
    gradients. Minimising their sum, the curve settles onto the object boundary,
    interpolating across gaps where the edge is weak. An optional ``balloon`` force
    inflates (positive) or deflates (negative) the curve along its normal so it
    reliably travels to a distant boundary instead of stalling in a flat region.
    Solved implicitly with the pentadiagonal internal-force matrix. ``init_snake``
    is an (N, 2) array of (row, col) points enclosing the object.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    gy, gx = np.gradient(img)
    edge = gx ** 2 + gy ** 2                              # external potential
    fy, fx = np.gradient(edge)                            # pulls toward strong edges
    snake = np.array(init_snake, float)
    n = len(snake)
    # pentadiagonal matrix A for internal (smoothness) forces on a closed curve
    a = np.roll(np.eye(n), -1, 0) + np.roll(np.eye(n), 1, 0) - 2 * np.eye(n)
    b = (np.roll(np.eye(n), -2, 0) + np.roll(np.eye(n), 2, 0)
         - 4 * np.roll(np.eye(n), -1, 0) - 4 * np.roll(np.eye(n), 1, 0)
         + 6 * np.eye(n))
    A = -alpha * a + beta * b
    inv = np.linalg.inv(np.eye(n) + gamma * A)
    for _ in range(n_iter):
        r = np.clip(snake[:, 0].astype(int), 0, img.shape[0] - 1)
        c = np.clip(snake[:, 1].astype(int), 0, img.shape[1] - 1)
        force = np.column_stack([fy[r, c], fx[r, c]]) * w_edge
        if balloon:
            centroid = snake.mean(axis=0)                 # inward/outward normal
            normal = centroid - snake
            normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9
            force = force + balloon * normal              # +: deflate toward centre
        snake = inv @ (snake + gamma * force)
        snake[:, 0] = np.clip(snake[:, 0], 0, img.shape[0] - 1)
        snake[:, 1] = np.clip(snake[:, 1], 0, img.shape[1] - 1)
    return snake


def orb_descriptors(image, keypoints=None, n_keypoints=50, patch=15, n_bits=256,
                    random_state=0):
    """Oriented FAST + rotated BRIEF: fast BINARY descriptors (Rublee, 2011).

    ORB describes a keypoint by a string of BITS, each the outcome of comparing two
    pixel intensities in its neighbourhood ("is point a brighter than point b?").
    Crucially the comparison pattern is ROTATED to the patch's dominant orientation
    (from the intensity centroid), so the descriptor is rotation-invariant, and
    matching is a Hamming distance -- an XOR and a bit-count, orders of magnitude
    faster than the float descriptors it replaced. Returns (keypoints, bit
    descriptors).
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    rng = check_random_state(random_state)
    if keypoints is None:
        resp = harris_corners(img)
        keypoints = corner_peaks(resp, min_distance=5)[:n_keypoints]
    half = patch // 2
    pairs = rng.randint(-half, half + 1, size=(n_bits, 4))   # (ay,ax,by,bx)
    desc, kept = [], []
    H, W = img.shape
    for (y, x) in keypoints:
        if not (half <= y < H - half and half <= x < W - half):
            continue
        win = img[y - half:y + half + 1, x - half:x + half + 1]
        # orientation from the intensity centroid
        ys, xs = np.mgrid[-half:half + 1, -half:half + 1]
        m00 = win.sum() + 1e-9
        theta = np.arctan2((ys * win).sum() / m00, (xs * win).sum() / m00)
        ct, st = np.cos(theta), np.sin(theta)
        bits = []
        for ay, ax, by, bx in pairs:
            ra = int(round(ay * ct - ax * st)); ca = int(round(ay * st + ax * ct))
            rb = int(round(by * ct - bx * st)); cb = int(round(by * st + bx * ct))
            pa = img[np.clip(y + ra, 0, H - 1), np.clip(x + ca, 0, W - 1)]
            pb = img[np.clip(y + rb, 0, H - 1), np.clip(x + cb, 0, W - 1)]
            bits.append(1 if pa < pb else 0)
        desc.append(bits); kept.append((y, x))
    return np.array(kept), np.array(desc, dtype=np.uint8)


def match_descriptors(desc1, desc2, max_ratio=0.8):
    """Match two sets of binary descriptors by Hamming distance + ratio test."""
    if len(desc1) == 0 or len(desc2) == 0:
        return np.empty((0, 2), int)
    # hamming distance matrix via XOR bit counts
    D = (desc1[:, None, :] != desc2[None, :, :]).sum(axis=2)
    matches = []
    for i in range(len(desc1)):
        order = np.argsort(D[i])
        best, second = D[i, order[0]], D[i, order[1]] if len(order) > 1 else 1e9
        if best < max_ratio * (second + 1e-9):           # Lowe's ratio test
            matches.append((i, int(order[0])))
    return np.array(matches) if matches else np.empty((0, 2), int)


class BagOfVisualWords(BaseEstimator):
    """Represent an image as a HISTOGRAM of quantised local features (Sivic, 2003).

    Borrowed from text: treat local image descriptors as "visual words". Cluster a
    corpus of descriptors into a VOCABULARY (k-means centres), then describe any
    image by the histogram of which visual words its descriptors fall into -- a
    fixed-length vector regardless of image size or keypoint count, ready for a
    plain classifier. Order and position are discarded, exactly like bag-of-words
    for documents.
    """

    def __init__(self, vocab_size=32, patch=8, stride=8, random_state=None):
        self.vocab_size = vocab_size
        self.patch = patch
        self.stride = stride
        self.random_state = random_state

    def _patches(self, image):
        img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
        p, s = self.patch, self.stride
        out = []
        for y in range(0, img.shape[0] - p + 1, s):
            for x in range(0, img.shape[1] - p + 1, s):
                patch = img[y:y + p, x:x + p].ravel()
                out.append(patch - patch.mean())         # contrast-normalise
        return np.array(out)

    def fit(self, images):
        from ..cluster import KMeans
        feats = np.vstack([self._patches(im) for im in images])
        self.kmeans_ = KMeans(n_clusters=self.vocab_size,
                              random_state=self.random_state).fit(feats)
        return self

    def transform(self, images):
        out = []
        for im in images:
            words = self.kmeans_.predict(self._patches(im))
            hist = np.bincount(words, minlength=self.vocab_size).astype(float)
            out.append(hist / (hist.sum() + 1e-9))       # normalised histogram
        return np.array(out)


def seam_carving(image, n_seams, axis=1):
    """Resize by removing the least important PATHS of pixels (Avidan, 2007).

    Scaling squashes everything equally; cropping loses the edges. Seam carving
    removes, one at a time, the connected path of pixels ("seam") with the least
    ENERGY (gradient magnitude), so it shrinks an image through its bland regions
    and leaves the salient objects intact. Each seam is the minimum-cost top-to-
    bottom path found by dynamic programming. Set ``axis=0`` to carve horizontal
    seams (reduce height). Returns the carved image.
    """
    img = np.asarray(image, float)
    if axis == 0:
        return seam_carving(img.T if img.ndim == 2 else img.transpose(1, 0, 2),
                            n_seams, axis=1).T if img.ndim == 2 else \
            seam_carving(img.transpose(1, 0, 2), n_seams, axis=1).transpose(1, 0, 2)
    for _ in range(n_seams):
        gray = _gray(img) if img.ndim == 3 else img
        gy, gx = np.gradient(gray)
        energy = np.abs(gx) + np.abs(gy)
        H, W = energy.shape
        cost = energy.copy()
        back = np.zeros((H, W), int)
        for y in range(1, H):                            # DP for the min seam
            for x in range(W):
                lo, hi = max(0, x - 1), min(W, x + 2)
                k = lo + int(np.argmin(cost[y - 1, lo:hi]))
                back[y, x] = k
                cost[y, x] += cost[y - 1, k]
        seam = np.empty(H, int)
        seam[-1] = int(np.argmin(cost[-1]))
        for y in range(H - 2, -1, -1):
            seam[y] = back[y + 1, seam[y + 1]]
        # remove the seam column-by-row
        keep = np.ones((H, W), bool)
        keep[np.arange(H), seam] = False
        if img.ndim == 3:
            img = np.stack([img[:, :, c][keep].reshape(H, W - 1)
                            for c in range(img.shape[2])], axis=2)
        else:
            img = img[keep].reshape(H, W - 1)
    return img


__all__ = ["gaussian_pyramid", "laplacian_pyramid", "pyramid_reconstruct",
           "watershed", "active_contour", "orb_descriptors", "match_descriptors",
           "BagOfVisualWords", "seam_carving"]
