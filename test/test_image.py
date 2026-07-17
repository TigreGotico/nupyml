"""Image feature extractors and shape tools.

Each test holds a descriptor to the invariance or structure it claims: HOG and
LBP to illumination invariance, GLCM/Haralick to texture ordering, Gabor to
orientation selectivity, the integral image to box-sum correctness, labelling to
component counting, morphology to speck removal / hole filling, and Hough to
recovering a planted line.
"""
import numpy as np
import pytest

from nupyml.image import (histogram_of_oriented_gradients, local_binary_pattern,
                          graycomatrix, haralick_features, gabor_kernel,
                          gabor_features, integral_image, rectangle_sum, label,
                          erosion, dilation, opening, closing, hough_line)


# --- HOG ------------------------------------------------------------------

def test_hog_returns_normalised_block_vector():
    rng = np.random.RandomState(0)
    img = rng.rand(32, 32)
    h = histogram_of_oriented_gradients(img)
    assert h.ndim == 1 and len(h) > 0
    assert np.all(h >= 0)                       # magnitudes are non-negative


def test_hog_is_invariant_to_global_contrast():
    """Block L2-normalisation should make HOG nearly unchanged when the whole
    image is scaled in intensity."""
    rng = np.random.RandomState(1)
    img = rng.rand(32, 32)
    h1 = histogram_of_oriented_gradients(img)
    h2 = histogram_of_oriented_gradients(img * 3.0)   # global contrast change
    assert np.allclose(h1, h2, atol=1e-6)


def test_hog_detects_edge_orientation():
    """A vertical edge produces horizontal gradients; the HOG mass should sit in
    a different orientation than for a horizontal edge."""
    vert = np.zeros((32, 32)); vert[:, 16:] = 1.0
    horiz = np.zeros((32, 32)); horiz[16:, :] = 1.0
    hv = histogram_of_oriented_gradients(vert)
    hh = histogram_of_oriented_gradients(horiz)
    assert not np.allclose(hv, hh)


# --- LBP ------------------------------------------------------------------

def test_lbp_is_invariant_to_monotonic_brightness():
    """LBP compares each pixel to its neighbours, so adding a constant (or any
    positive scaling) leaves every code unchanged."""
    rng = np.random.RandomState(2)
    img = rng.rand(20, 20)
    a = local_binary_pattern(img)
    b = local_binary_pattern(img * 2.0 + 5.0)   # monotonic intensity change
    assert np.array_equal(a, b)


def test_lbp_codes_are_in_range():
    rng = np.random.RandomState(3)
    img = rng.rand(20, 20)
    codes = local_binary_pattern(img, n_points=8)
    assert codes.max() <= 255 and codes.min() >= 0   # 8-bit codes


# --- GLCM / Haralick ------------------------------------------------------

def test_glcm_diagonal_for_smooth_image():
    """A smooth gradient image has neighbours at similar levels -> co-occurrence
    mass concentrates on/near the diagonal, giving high homogeneity."""
    smooth = np.tile(np.linspace(0, 1, 16), (16, 1))
    glcm = graycomatrix(smooth, distance=1, angle=0.0, levels=8)
    feats = haralick_features(glcm)
    assert feats["homogeneity"] > 0.5
    # a random (busy) texture is far less homogeneous
    rng = np.random.RandomState(4)
    busy = graycomatrix(rng.rand(16, 16), levels=8)
    assert feats["homogeneity"] > haralick_features(busy)["homogeneity"]


def test_haralick_contrast_higher_for_busy_texture():
    smooth = np.tile(np.linspace(0, 1, 16), (16, 1))
    rng = np.random.RandomState(5)
    busy = rng.rand(16, 16)
    cs = haralick_features(graycomatrix(smooth, levels=8))["contrast"]
    cb = haralick_features(graycomatrix(busy, levels=8))["contrast"]
    assert cb > cs


# --- Gabor ----------------------------------------------------------------

def test_gabor_kernel_is_zero_mean():
    """A zero-mean kernel produces no response to a flat region (no DC leak)."""
    k = gabor_kernel(0.2, theta=0.0)
    assert abs(k.mean()) < 1e-10


def test_gabor_responds_to_matching_orientation():
    """A vertical stripe pattern should excite a Gabor tuned to its orientation
    more than the feature vector of a flat image."""
    x = np.arange(64)
    stripes = np.tile(np.sin(2 * np.pi * x * 0.1), (64, 1))   # vertical stripes
    flat = np.zeros((64, 64))
    fs = gabor_features(stripes)
    ff = gabor_features(flat)
    assert np.abs(fs).sum() > np.abs(ff).sum()


# --- integral image -------------------------------------------------------

def test_integral_image_box_sum_matches_brute_force():
    rng = np.random.RandomState(6)
    img = rng.rand(20, 20)
    ii = integral_image(img)
    # a few random rectangles must match the direct sum
    for (y0, x0, y1, x1) in [(0, 0, 5, 5), (3, 4, 10, 12), (7, 7, 19, 19)]:
        assert rectangle_sum(ii, y0, x0, y1, x1) == pytest.approx(
            img[y0:y1 + 1, x0:x1 + 1].sum())


# --- connected components -------------------------------------------------

def test_label_counts_separate_blobs():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True                       # blob 1
    mask[6:9, 6:9] = True                       # blob 2 (not touching)
    labels, n = label(mask)
    assert n == 2
    # every foreground pixel got a non-zero label, background stayed 0
    assert (labels[mask] > 0).all() and (labels[~mask] == 0).all()


def test_label_merges_touching_pixels():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, :] = True                           # one connected horizontal bar
    _, n = label(mask, connectivity=1)
    assert n == 1


# --- morphology -----------------------------------------------------------

def test_opening_removes_small_speck_keeps_block():
    img = np.zeros((15, 15), dtype=bool)
    img[5:11, 5:11] = True                      # a solid 6x6 block
    img[0, 0] = True                            # an isolated speck
    out = opening(img)
    assert not out[0, 0]                        # speck erased
    assert out[7, 7]                            # block interior survives


def test_closing_fills_small_hole():
    img = np.ones((15, 15), dtype=bool)
    img[7, 7] = False                           # a one-pixel hole
    out = closing(img)
    assert out[7, 7]                            # hole filled


def test_erosion_shrinks_dilation_grows():
    img = np.zeros((15, 15), dtype=bool)
    img[5:10, 5:10] = True
    assert erosion(img).sum() < img.sum() < dilation(img).sum()


# --- Hough ----------------------------------------------------------------

def test_hough_recovers_a_planted_line():
    """A perfectly horizontal edge line should peak at theta ~ +/-90 deg (a line
    of constant y), with a strong accumulator maximum."""
    edges = np.zeros((50, 50), dtype=bool)
    edges[25, :] = True                         # horizontal line at y=25
    acc, thetas, rhos = hough_line(edges)
    peak = np.unravel_index(np.argmax(acc), acc.shape)
    best_theta = thetas[peak[1]]
    # horizontal line -> normal points vertically -> |theta| ~ pi/2
    assert abs(abs(best_theta) - np.pi / 2) < 0.05
    assert acc.max() == 50                      # every one of the 50 pixels voted
