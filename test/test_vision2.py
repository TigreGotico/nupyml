"""J4: vision v2 -- pyramids, watershed, active contours, ORB, bag-of-visual-words,
seam carving.

The Laplacian pyramid inverts exactly; watershed floods seeded basins to segment
touching blobs; the balloon snake contracts onto an object boundary; ORB matches an
image to a shifted copy via Hamming distance; bag-of-visual-words yields normalised
histograms; seam carving removes exactly N columns.
"""
import numpy as np
import scipy.ndimage as ndi
import pytest

from nupyml.image import (gaussian_pyramid, laplacian_pyramid, pyramid_reconstruct,
                          watershed, active_contour, orb_descriptors,
                          match_descriptors, BagOfVisualWords, seam_carving)


def test_gaussian_pyramid_halves_resolution():
    img = np.random.RandomState(0).rand(64, 64)
    pyr = gaussian_pyramid(img, levels=4)
    assert [g.shape for g in pyr] == [(64, 64), (32, 32), (16, 16), (8, 8)]


def test_laplacian_pyramid_reconstructs_exactly():
    img = np.random.RandomState(0).rand(64, 64)
    lap = laplacian_pyramid(img, levels=4)
    rec = pyramid_reconstruct(lap)
    assert np.allclose(rec, img, atol=1e-6)


def test_watershed_separates_two_basins():
    yy, xx = np.mgrid[0:40, 0:40]
    field = (np.exp(-((yy - 10) ** 2 + (xx - 10) ** 2) / 50)
             + np.exp(-((yy - 30) ** 2 + (xx - 30) ** 2) / 50))
    markers = np.zeros((40, 40), int)
    markers[10, 10] = 1; markers[30, 30] = 2
    labels = watershed(-field, markers)                  # flood the inverted field
    assert set(np.unique(labels)) == {1, 2}
    assert labels[10, 10] == 1 and labels[30, 30] == 2
    assert labels[12, 12] == 1 and labels[28, 28] == 2   # basins claim their blob


def test_active_contour_contracts_onto_object():
    im = np.zeros((80, 80)); im[30:50, 30:50] = 1.0
    im = ndi.gaussian_filter(im, 3)
    t = np.linspace(0, 2 * np.pi, 40)
    init = np.column_stack([40 + 30 * np.sin(t), 40 + 30 * np.cos(t)])
    snake = active_contour(im, init, alpha=0.01, beta=0.05, gamma=0.3,
                           w_edge=200.0, balloon=0.2, n_iter=500)
    r = np.mean(np.sqrt((snake[:, 0] - 40) ** 2 + (snake[:, 1] - 40) ** 2))
    assert 6 < r < 18                                     # contracted from 30 onto the square


def test_orb_matches_shifted_image():
    img = np.random.RandomState(0).rand(64, 64)
    shifted = np.roll(img, 3, axis=1)
    kp1, d1 = orb_descriptors(img, n_keypoints=40)
    kp2, d2 = orb_descriptors(shifted, n_keypoints=40)
    assert d1.shape[1] == 256                             # 256-bit descriptors
    matches = match_descriptors(d1, d2)
    assert len(matches) > 3                               # finds correspondences


def test_bag_of_visual_words_histograms():
    rng = np.random.RandomState(0)
    imgs = [rng.rand(32, 32) for _ in range(6)]
    bovw = BagOfVisualWords(vocab_size=8, random_state=0).fit(imgs)
    H = bovw.transform(imgs)
    assert H.shape == (6, 8)
    assert np.allclose(H.sum(axis=1), 1.0)               # normalised histograms


def test_seam_carving_removes_exact_columns():
    img = np.random.RandomState(0).rand(48, 48)
    carved = seam_carving(img, n_seams=10, axis=1)
    assert carved.shape == (48, 38)                      # 10 vertical seams removed
    tall = seam_carving(img, n_seams=5, axis=0)
    assert tall.shape == (43, 48)                        # 5 horizontal seams removed
