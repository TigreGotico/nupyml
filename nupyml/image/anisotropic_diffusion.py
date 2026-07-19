"""Anisotropic diffusion (Perona-Malik): diffusion that stops at edges."""
import numpy as np

from .vision import _gray


def anisotropic_diffusion(image, n_iter=20, kappa=0.1, gamma=0.2):
    """Diffusion that STOPS at edges (Perona & Malik, 1990).

    Ordinary (isotropic) diffusion is exactly Gaussian blur -- it smooths everywhere,
    edges included. Perona-Malik makes the diffusion CONDUCTANCE depend on the local
    gradient: heat flows freely within smooth regions but is BLOCKED where the gradient
    is large (an edge). So noise diffuses away inside regions while edges are preserved
    and even sharpened. ``kappa`` sets the gradient scale at which diffusion stops;
    ``gamma`` is the time step.
    """
    img = _gray(image) if np.asarray(image).ndim == 3 else np.asarray(image, float)
    u = img.copy()
    for _ in range(n_iter):
        dn = np.roll(u, -1, 0) - u; ds = np.roll(u, 1, 0) - u
        de = np.roll(u, -1, 1) - u; dw = np.roll(u, 1, 1) - u
        # edge-stopping conductance: exp(-(grad/kappa)^2), small at strong edges
        cn = np.exp(-(dn / kappa) ** 2); cs = np.exp(-(ds / kappa) ** 2)
        ce = np.exp(-(de / kappa) ** 2); cw = np.exp(-(dw / kappa) ** 2)
        u = u + gamma * (cn * dn + cs * ds + ce * de + cw * dw)
    return u


__all__ = ["anisotropic_diffusion"]
