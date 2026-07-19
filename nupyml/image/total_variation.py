"""Total-variation (ROF) denoising: remove noise by making the gradient sparse."""
import numpy as np

from .vision import _gray


def total_variation_denoise(image, weight=0.1, n_iter=100, step=0.2):
    """Denoise by making the gradient SPARSE (Rudin-Osher-Fatemi, 1992).

    Total-variation denoising minimises ``||u - f||^2 + weight * TV(u)`` where TV is the
    integral of the gradient magnitude. Penalising the L1 of the gradient (not the L2)
    is the key: it removes noise -- whose gradient is everywhere -- while ALLOWING the
    few large jumps of true edges, because an L1 penalty tolerates sparse large values.
    The result is piecewise-smooth with crisp edges, the "cartoon" of the image.
    Gradient-descent on the ROF energy here.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    u = img.copy()
    for _ in range(n_iter):
        gx = np.diff(u, axis=1, append=u[:, -1:])
        gy = np.diff(u, axis=0, append=u[-1:, :])
        gmag = np.sqrt(gx ** 2 + gy ** 2) + 1e-8
        # divergence of the normalised gradient (curvature) drives the TV term
        div = (np.diff(gx / gmag, axis=1, prepend=(gx / gmag)[:, :1])
               + np.diff(gy / gmag, axis=0, prepend=(gy / gmag)[:1, :]))
        u = u - step * ((u - img) - weight * div)
    return u


__all__ = ["total_variation_denoise"]
