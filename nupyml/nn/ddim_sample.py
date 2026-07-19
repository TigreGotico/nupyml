"""Sample a trained diffusion model DETERMINISTICALLY, in few steps"""
import numpy as np
from ..autograd import Tensor
from ..utils import check_random_state


def ddim_sample(ddpm, n, n_steps=20, eta=0.0, rng=None):
    """Sample a trained diffusion model DETERMINISTICALLY, in few steps
    (Song et al., 2021).

    DDPM sampling is stochastic and needs all ``T`` denoising steps. DDIM reuses the
    SAME trained noise predictor but follows a non-Markovian, (with ``eta=0``)
    fully DETERMINISTIC trajectory: at each step it predicts ``x_0`` from the current
    noisy sample and jumps directly toward it along a chosen subsequence of
    timesteps. This lets you sample in, say, 20 steps instead of 1000, and -- being
    deterministic -- gives a meaningful latent-to-image map. ``eta`` interpolates
    back toward stochastic DDPM.
    """
    rng = check_random_state(rng)
    ab = ddpm.alpha_bars
    d = ddpm.model.net  # noise predictor MLP  (via ddpm.model)
    x = rng.normal(size=(n, ddpm.n_features))
    ts = np.linspace(ddpm.T - 1, 0, n_steps).astype(int)
    for i, t in enumerate(ts):
        eps = ddpm.model(Tensor(x), np.full(n, t)).data
        x0 = (x - np.sqrt(1 - ab[t]) * eps) / np.sqrt(ab[t])   # predict clean sample
        if i == len(ts) - 1:
            x = x0
            break
        t_prev = ts[i + 1]
        sigma = eta * np.sqrt((1 - ab[t_prev]) / (1 - ab[t]) *
                              (1 - ab[t] / ab[t_prev]))
        noise = rng.normal(size=x.shape) if eta > 0 else 0.0
        x = (np.sqrt(ab[t_prev]) * x0
             + np.sqrt(np.maximum(1 - ab[t_prev] - sigma ** 2, 0)) * eps
             + sigma * noise)
    return x


__all__ = ["ddim_sample"]
