"""L4: vision v4 -- FAST corners, LoG blobs, Hough circles, fundamental matrix,
mean-shift segmentation.

FAST fires at a square's corners; blob detection finds two disks and reads off
their radii; Hough circles locate a circle's centre; the 8-point algorithm recovers
a fundamental matrix satisfying the epipolar constraint; mean-shift segments a
four-region image into four segments.
"""
import numpy as np
import scipy.ndimage as ndi
import pytest

from nupyml.image import (fast_corners, blob_detection, hough_circles,
                          fundamental_matrix, ransac_fundamental,
                          MeanShiftSegmentation)


def test_fast_detects_square_corners():
    img = np.zeros((40, 40)); img[12:28, 12:28] = 1.0
    corners = fast_corners(img, threshold=0.3, n_contiguous=9)
    assert len(corners) > 0
    # at least one corner sits near the square's top-left corner
    assert any(abs(c[0] - 12) < 3 and abs(c[1] - 12) < 3 for c in corners)


def test_blob_detection_finds_two_sizes():
    im = np.zeros((60, 60))
    yy, xx = np.mgrid[0:60, 0:60]
    im[(yy - 20) ** 2 + (xx - 20) ** 2 < 25] = 1.0        # radius 5
    im[(yy - 40) ** 2 + (xx - 40) ** 2 < 64] = 1.0        # radius 8
    im = ndi.gaussian_filter(im, 1)
    blobs = blob_detection(im, min_sigma=2, max_sigma=10, threshold=0.02)
    assert len(blobs) == 2
    radii = sorted(b[2] for b in blobs)
    assert radii[0] < radii[1]                            # a small and a large blob


def test_hough_circles_finds_the_centre():
    c = np.zeros((60, 60))
    yy, xx = np.mgrid[0:60, 0:60]
    c[(yy - 30) ** 2 + (xx - 30) ** 2 <= 100] = 1.0       # a disk of radius 10
    circles = hough_circles(c, radii=[8, 10, 12], threshold=0.35)
    assert len(circles) > 0
    assert any(np.hypot(y - 30, x - 30) < 5 for y, x, r in circles)


def test_fundamental_matrix_satisfies_epipolar_constraint():
    rng = np.random.RandomState(0)
    F = np.array([[0, -0.3, 0.1], [0.3, 0, -0.2], [-0.05, 0.15, 0.02]])
    U, S, Vt = np.linalg.svd(F); S[2] = 0; F = U @ np.diag(S) @ Vt
    p1 = rng.rand(30, 2) * 100
    lines = (F @ np.column_stack([p1, np.ones(30)]).T).T
    p2 = np.array([[(x := rng.rand() * 100), (-c0 - a * x) / (b + 1e-9)]
                   for a, b, c0 in lines])
    Fest = fundamental_matrix(p1, p2)
    resid = np.mean([abs(np.array([*p2[i], 1]) @ Fest @ np.array([*p1[i], 1]))
                     for i in range(30)])
    assert resid < 1e-3
    assert np.linalg.matrix_rank(Fest, tol=1e-6) == 2    # F is rank 2


def test_ransac_fundamental_rejects_outliers():
    rng = np.random.RandomState(1)
    F = np.array([[0, -0.3, 0.1], [0.3, 0, -0.2], [-0.05, 0.15, 0.02]])
    U, S, Vt = np.linalg.svd(F); S[2] = 0; F = U @ np.diag(S) @ Vt
    p1 = rng.rand(40, 2) * 100
    lines = (F @ np.column_stack([p1, np.ones(40)]).T).T
    p2 = np.array([[(x := rng.rand() * 100), (-c0 - a * x) / (b + 1e-9)]
                   for a, b, c0 in lines])
    p2[::5] = rng.rand(8, 2) * 100                        # 8 outlier matches
    Fr, inliers = ransac_fundamental(p1, p2, threshold=0.1, n_iter=800,
                                     random_state=0)
    assert inliers.sum() >= 28                            # keeps the clean matches
    assert not inliers[::5].any()                         # rejects the outliers


def test_mean_shift_segments_four_regions():
    seg = np.zeros((20, 20))
    seg[:10, :10] = 0.2; seg[:10, 10:] = 0.5
    seg[10:, :10] = 0.8; seg[10:, 10:] = 0.35
    ms = MeanShiftSegmentation(spatial_radius=10, color_radius=0.15, max_iter=25)
    labels = ms.fit_predict(seg)
    assert len(np.unique(labels)) == 4
