"""The epipolar relation between two views: the normalised 8-POINT algorithm"""
import numpy as np


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


__all__ = ["fundamental_matrix"]
