"""The 3x3 projective map from >=4 point correspondences, via the DLT."""
import numpy as np


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


__all__ = ["estimate_homography"]
