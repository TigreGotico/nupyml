"""Fit the fundamental matrix robustly to outlier correspondences."""
import numpy as np
from ..utils import check_random_state
from .fundamental_matrix import fundamental_matrix


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


__all__ = ["ransac_fundamental"]
