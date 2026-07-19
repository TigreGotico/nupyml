"""DENSE optical flow with a global smoothness prior (Horn & Schunck, 1981)."""
import numpy as np
import scipy.ndimage as ndi
from .felzenszwalb import felzenszwalb


def horn_schunck(frame1, frame2, alpha=1.0, n_iter=100):
    """DENSE optical flow with a global smoothness prior (Horn & Schunck, 1981).

    Lucas-Kanade estimates flow only at good corners. Horn-Schunck instead solves
    for a flow vector at EVERY pixel by minimising, globally, the brightness-
    constancy error plus a SMOOTHNESS penalty (neighbouring flow vectors should
    agree). ``alpha`` weights the smoothness: larger gives a smoother, more filled-in
    field, which is what propagates motion into the textureless regions Lucas-Kanade
    leaves blank. Solved by the classic Jacobi iteration. Returns the ``(u, v)`` flow
    fields.
    """
    f1 = np.asarray(frame1, float); f2 = np.asarray(frame2, float)
    Ix = ndi.sobel(f1, axis=1) / 8.0
    Iy = ndi.sobel(f1, axis=0) / 8.0
    It = f2 - f1
    u = np.zeros_like(f1); v = np.zeros_like(f1)
    kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                       [1 / 6, 0, 1 / 6],
                       [1 / 12, 1 / 6, 1 / 12]])
    for _ in range(n_iter):
        u_bar = ndi.convolve(u, kernel, mode="nearest")
        v_bar = ndi.convolve(v, kernel, mode="nearest")
        denom = alpha ** 2 + Ix ** 2 + Iy ** 2
        common = (Ix * u_bar + Iy * v_bar + It) / denom
        u = u_bar - Ix * common                          # Jacobi update
        v = v_bar - Iy * common
    return u, v


# --------------------------------------------------------------- felzenszwalb


__all__ = ["horn_schunck"]
