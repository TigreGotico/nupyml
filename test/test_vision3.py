"""K4: vision v3 -- SIFT, Horn-Schunck, Felzenszwalb, homography+RANSAC, Viola-Jones.

SIFT returns normalised 128-D descriptors at scale-space extrema; Horn-Schunck
recovers the direction of a global translation; Felzenszwalb splits a two-region
image; the DLT recovers a known homography exactly and RANSAC rejects outliers;
Viola-Jones learns to detect a bright-centre pattern.
"""
import numpy as np
import scipy.ndimage as ndi
import pytest

from nupyml.image import (sift, horn_schunck, felzenszwalb, estimate_homography,
                          ransac_homography, ViolaJones)
from nupyml.metrics import accuracy_score


def test_sift_descriptors_are_128d_and_normalised():
    rng = np.random.RandomState(0)
    blob = ndi.gaussian_filter((rng.rand(64, 64) > 0.7).astype(float), 2)
    kp, desc = sift(blob, n_keypoints=50)
    assert len(kp) > 0
    assert desc.shape[1] == 128
    assert np.allclose(np.linalg.norm(desc, axis=1), 1.0, atol=0.1)


def test_horn_schunck_recovers_translation_direction():
    rng = np.random.RandomState(0)
    base = ndi.gaussian_filter(rng.rand(40, 40), 2.0)
    shifted = np.roll(base, 1, axis=1)                  # move content right
    u, v = horn_schunck(base, shifted, alpha=0.5, n_iter=300)
    assert u[10:30, 10:30].mean() > 0                   # horizontal flow, right
    assert abs(u.mean()) > abs(v.mean())                # mostly horizontal


def test_felzenszwalb_splits_two_regions():
    rng = np.random.RandomState(0)
    img = np.zeros((40, 40)); img[:, :20] = 0.2; img[:, 20:] = 0.8
    img += rng.randn(40, 40) * 0.02
    labels = felzenszwalb(img, scale=50, sigma=0.5, min_size=30)
    assert len(np.unique(labels)) == 2


def test_homography_dlt_is_exact():
    rng = np.random.RandomState(0)
    H = np.array([[1.1, 0.1, 5], [0.05, 1.2, 3], [0.0001, 0.0002, 1.0]])
    src = rng.rand(4, 2) * 50
    p = np.column_stack([src, np.ones(4)]) @ H.T
    dst = p[:, :2] / p[:, 2:3]
    Hest = estimate_homography(src, dst)
    assert np.allclose(Hest / Hest[2, 2], H / H[2, 2], atol=1e-6)


def test_ransac_homography_rejects_outliers():
    rng = np.random.RandomState(0)
    H = np.array([[1.1, 0.1, 5], [0.05, 1.2, 3], [0.0001, 0.0002, 1.0]])
    src = rng.rand(20, 2) * 50
    p = np.column_stack([src, np.ones(20)]) @ H.T
    dst = p[:, :2] / p[:, 2:3]
    dst[::5] += rng.rand(4, 2) * 100                    # 4 gross outliers
    Hr, inliers = ransac_homography(src, dst, threshold=2.0, n_iter=500,
                                    random_state=0)
    assert inliers.sum() >= 15                          # the 16 clean matches
    assert not inliers[::5].any()                       # outliers excluded
    assert np.allclose(Hr / Hr[2, 2], H / H[2, 2], atol=0.1)


def test_viola_jones_detects_bright_center():
    rng = np.random.RandomState(0)

    def make(bright):
        im = rng.rand(24, 24) * 0.3
        if bright:
            im[8:16, 8:16] += 0.7
        return im

    Xtr = [make(i % 2 == 0) for i in range(60)]
    ytr = [1 if i % 2 == 0 else 0 for i in range(60)]
    vj = ViolaJones(window=24, n_rounds=20, step=4).fit(Xtr, ytr)
    Xte = [make(i % 2 == 0) for i in range(40)]
    yte = [1 if i % 2 == 0 else 0 for i in range(40)]
    assert accuracy_score(yte, vj.predict(Xte)) > 0.9
