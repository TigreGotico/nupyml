"""Classic computer-vision operators: corners, edges, matching, flow, distance,
superpixels, and non-max suppression.

Where ``features`` (HOG/LBP/GLCM/Gabor) describes whole patches and ``shape`` does
morphology, these are the geometric primitives of a vision pipeline -- finding
WHERE the interesting points and edges are, tracking them, and grouping pixels.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, sobel


def _gray(image):
    image = np.asarray(image, float)
    return image.mean(axis=2) if image.ndim == 3 else image


def harris_corners(image, k=0.04, sigma=1.0, threshold=0.01):
    """Harris corner detector: find points where intensity varies in BOTH directions.

    THE IDEA
    --------
    A flat region looks the same when you shift a window; an edge looks the same
    ALONG the edge; only a CORNER changes under a shift in every direction. Harris
    quantifies this from the STRUCTURE TENSOR -- the smoothed outer product of the
    image gradients over a window. Its two eigenvalues measure gradient strength in
    two directions; both large means a corner. The response
    ``R = det(M) - k*trace(M)^2`` is large and positive exactly there, avoiding an
    explicit (costly) eigendecomposition. Returns the response map; the peaks are
    the corners.
    """
    g = _gray(image)
    Ix = sobel(g, axis=1)
    Iy = sobel(g, axis=0)
    Sxx = gaussian_filter(Ix * Ix, sigma)
    Syy = gaussian_filter(Iy * Iy, sigma)
    Sxy = gaussian_filter(Ix * Iy, sigma)
    det = Sxx * Syy - Sxy ** 2
    trace = Sxx + Syy
    R = det - k * trace ** 2
    return R


def corner_peaks(response, min_distance=3, threshold_rel=0.1):
    """Corner coordinates from a Harris response: threshold + local-max NMS."""
    from scipy.ndimage import maximum_filter
    thr = threshold_rel * response.max()
    local_max = (response == maximum_filter(response, size=2 * min_distance + 1))
    ys, xs = np.where(local_max & (response > thr))
    return np.column_stack([ys, xs])


def canny(image, sigma=1.0, low=0.1, high=0.2):
    """Canny edge detector: thin, connected edges via NMS + hysteresis.

    A gradient magnitude alone gives thick, noisy edges. Canny adds the two ideas
    that made it the standard: NON-MAXIMUM SUPPRESSION thins each ridge to a
    one-pixel line by keeping only pixels that are a local maximum ACROSS the
    gradient direction; and HYSTERESIS thresholding keeps strong edges (above
    ``high``) plus weak edges (above ``low``) only if they CONNECT to a strong one
    -- so real faint edges survive while isolated noise does not. Returns a binary
    edge map.
    """
    from scipy.ndimage import label as cc_label
    g = gaussian_filter(_gray(image), sigma)
    Ix, Iy = sobel(g, axis=1), sobel(g, axis=0)
    mag = np.hypot(Ix, Iy)
    mag /= (mag.max() + 1e-12)
    ang = (np.rad2deg(np.arctan2(Iy, Ix)) % 180)
    # non-maximum suppression across the gradient direction (4 quantised angles)
    thin = np.zeros_like(mag)
    H, W = mag.shape
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = ang[i, j]
            if a < 22.5 or a >= 157.5:
                n1, n2 = mag[i, j - 1], mag[i, j + 1]
            elif a < 67.5:
                n1, n2 = mag[i - 1, j + 1], mag[i + 1, j - 1]
            elif a < 112.5:
                n1, n2 = mag[i - 1, j], mag[i + 1, j]
            else:
                n1, n2 = mag[i - 1, j - 1], mag[i + 1, j + 1]
            if mag[i, j] >= n1 and mag[i, j] >= n2:
                thin[i, j] = mag[i, j]
    strong = thin >= high
    weak = (thin >= low) & ~strong
    # hysteresis: keep weak edges in a component that touches a strong edge
    lbl, n = cc_label(thin >= low)
    keep = set(lbl[strong].tolist())
    return np.isin(lbl, list(keep)) & (lbl > 0)


def match_template(image, template):
    """Locate a template by NORMALISED cross-correlation.

    Sliding the template over the image and correlating finds where it matches --
    but raw correlation is fooled by bright regions. NORMALISED cross-correlation
    subtracts the local mean and divides by the local std of both patches, so it
    measures SHAPE agreement independent of brightness and contrast. The peak of
    the returned correlation map is the best match location.
    """
    img = _gray(image)
    tmpl = _gray(template)
    th, tw = tmpl.shape
    t = tmpl - tmpl.mean()
    t_norm = np.sqrt((t ** 2).sum()) + 1e-12
    H, W = img.shape
    out = np.full((H - th + 1, W - tw + 1), -1.0)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = img[i:i + th, j:j + tw]
            p = patch - patch.mean()
            out[i, j] = (p * t).sum() / (np.sqrt((p ** 2).sum()) * t_norm + 1e-12)
    return out


def lucas_kanade(img1, img2, points, window=5):
    """Sparse Lucas-Kanade optical flow: track points between two frames.

    THE ASSUMPTION AND THE TRICK
    ----------------------------
    A point's brightness is constant between frames, so ``Ix*u + Iy*v + It = 0``
    -- one equation, two unknowns (the flow ``(u, v)``). Lucas-Kanade resolves the
    ambiguity by assuming the flow is CONSTANT over a small WINDOW, giving many
    equations for the same ``(u, v)`` and solving them by least squares. It tracks
    corners well (where the window's gradients span both directions) and fails on
    edges/flat regions (the aperture problem). Returns a flow vector per input
    point.
    """
    g1, g2 = _gray(img1), _gray(img2)
    Ix = sobel(g1, axis=1) / 8.0
    Iy = sobel(g1, axis=0) / 8.0
    It = g2 - g1
    w = window // 2
    flow = np.zeros((len(points), 2))
    for k, (y, x) in enumerate(np.asarray(points, int)):
        y0, y1 = max(0, y - w), min(g1.shape[0], y + w + 1)
        x0, x1 = max(0, x - w), min(g1.shape[1], x + w + 1)
        A = np.column_stack([Ix[y0:y1, x0:x1].ravel(), Iy[y0:y1, x0:x1].ravel()])
        b = -It[y0:y1, x0:x1].ravel()
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)   # least-squares (u, v)
        flow[k] = sol
    return flow


def chamfer_distance_transform(binary):
    """Distance from each pixel to the nearest FOREGROUND pixel (two-pass chamfer).

    The distance transform labels every background pixel with its distance to the
    nearest object -- the basis of skeletonisation, watershed markers, and shape
    matching. The chamfer method approximates the Euclidean distance in TWO raster
    passes (top-left to bottom-right, then back), each propagating the running
    minimum from already-visited neighbours with local step costs (1 orthogonally,
    sqrt(2) diagonally). Linear time, no distance ever computed globally.
    """
    b = np.asarray(binary, bool)
    D = np.where(b, 0.0, np.inf)
    d1, d2 = 1.0, np.sqrt(2)
    H, W = D.shape
    for i in range(H):                                 # forward pass
        for j in range(W):
            if D[i, j] == 0:
                continue
            for di, dj, c in [(-1, 0, d1), (0, -1, d1), (-1, -1, d2), (-1, 1, d2)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    D[i, j] = min(D[i, j], D[ni, nj] + c)
    for i in range(H - 1, -1, -1):                     # backward pass
        for j in range(W - 1, -1, -1):
            if D[i, j] == 0:
                continue
            for di, dj, c in [(1, 0, d1), (0, 1, d1), (1, 1, d2), (1, -1, d2)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    D[i, j] = min(D[i, j], D[ni, nj] + c)
    return D


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """Greedy NMS: keep the highest-scoring box, drop those that overlap it.

    A detector fires many overlapping boxes on one object. NMS keeps the
    highest-scoring box, removes every remaining box whose intersection-over-union
    with it exceeds ``iou_threshold``, and repeats -- collapsing each cluster of
    detections to one. ``boxes`` are ``(x1, y1, x2, y2)``; returns the kept indices.
    """
    boxes = np.asarray(boxes, float)
    scores = np.asarray(scores, float)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        order = order[1:][iou <= iou_threshold]        # drop the overlapping ones
    return keep


def slic(image, n_segments=50, compactness=10.0, n_iter=10):
    """SLIC superpixels: cluster pixels into compact, uniform regions
    (Achanta et al., 2012).

    Superpixels group pixels into perceptually-coherent regions, shrinking a
    million pixels to a few hundred primitives for downstream vision. SLIC is
    k-means in a joint COLOUR+POSITION space, but each pixel is only compared to
    cluster centres in its LOCAL neighbourhood (not the whole image), which is what
    makes it fast and the superpixels spatially compact. ``compactness`` trades
    colour similarity against spatial regularity. Returns a label per pixel.
    """
    img = _gray(image).astype(float)
    H, W = img.shape
    S = int(np.sqrt(H * W / n_segments))               # grid step
    # initialise centres on a regular grid: (y, x, intensity)
    centers = []
    for y in range(S // 2, H, S):
        for x in range(S // 2, W, S):
            centers.append([y, x, img[y, x]])
    centers = np.array(centers, float)
    labels = -np.ones((H, W), int)
    for _ in range(n_iter):
        dist = np.full((H, W), np.inf)
        for c, (cy, cx, ci) in enumerate(centers):
            y0, y1 = int(max(0, cy - S)), int(min(H, cy + S))
            x0, x1 = int(max(0, cx - S)), int(min(W, cx + S))
            ys, xs = np.mgrid[y0:y1, x0:x1]
            dc = (img[y0:y1, x0:x1] - ci) ** 2
            ds = (ys - cy) ** 2 + (xs - cx) ** 2       # spatial distance
            D = dc + (compactness / S) ** 2 * ds
            closer = D < dist[y0:y1, x0:x1]
            dist[y0:y1, x0:x1][closer] = D[closer]
            labels[y0:y1, x0:x1][closer] = c
        for c in range(len(centers)):                  # recompute centres
            mask = labels == c
            if mask.any():
                ys, xs = np.where(mask)
                centers[c] = [ys.mean(), xs.mean(), img[mask].mean()]
    return labels


__all__ = ["harris_corners", "corner_peaks", "canny", "match_template",
           "lucas_kanade", "chamfer_distance_transform", "non_max_suppression",
           "slic"]
