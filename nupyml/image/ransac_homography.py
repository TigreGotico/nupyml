"""Fit a homography ROBUSTLY, rejecting outlier matches (Fischler & Bolles, 1981)."""
import numpy as np
from ..utils import check_random_state
from .estimate_homography import estimate_homography


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


__all__ = ["ransac_homography"]
