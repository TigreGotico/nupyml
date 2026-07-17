"""Diffusion models: destroy the data slowly, then learn to undo one step.

THE IDEA
--------
Every other generative model tries to leap from noise to data in one go, and the
difficulty of that leap is what makes them hard to train. Diffusion refuses the
leap. It defines a FORWARD process that gradually adds gaussian noise until the
data is indistinguishable from N(0, I)::

    x_0 (data) -> x_1 -> x_2 -> ... -> x_T (pure noise)

That direction is fixed, known, and has no parameters -- nothing is learned. Then
it learns only to reverse ONE step at a time. Each step is a small, local
denoising problem, and a thousand easy problems beat one impossible one.

THE FIRST TRICK: JUMP TO ANY t IN ONE SHOT
------------------------------------------
Adding gaussian noise T times, naively, would need a loop of T steps for every
training sample. But a sum of gaussians is gaussian, so the composition has a
closed form::

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)

where ``alpha_bar_t`` is the running product of ``1 - beta``. So training samples
a random ``t``, jumps straight there, and never simulates the chain at all. This
is what makes training O(1) per sample instead of O(T).

Notice the interpolation: ``sqrt(alpha_bar)`` fades the signal out while
``sqrt(1 - alpha_bar)`` fades the noise in, and the coefficients are chosen so the
variance stays 1 throughout. The data never blows up; it just dissolves.

THE SECOND TRICK: PREDICT THE NOISE, NOT THE DATA
-------------------------------------------------
The reverse step needs the posterior mean, and the algebra can be arranged to ask
the network for any of three equivalent things: the clean ``x_0``, the posterior
mean directly, or the NOISE ``eps`` that was added. They are mathematically
interchangeable -- given any one, the other two follow.

Predicting ``eps`` wins in practice, and the loss is why::

    loss = || eps - eps_theta(x_t, t) ||^2

That is plain mean squared error against a target that is ALWAYS unit gaussian,
at every noise level. The target's scale never changes with ``t``, so no
reweighting across timesteps is needed and every step contributes comparably.
Predicting ``x_0`` instead gives a target whose difficulty varies wildly with
``t`` -- trivial at t=0, hopeless at t=T.

So the entire training objective of a diffusion model is: add noise, ask the
network which noise you added, take the squared error. It is startlingly simple,
and it took the ELBO of a T-step latent variable model to justify it.

WHY THE NETWORK MUST BE TOLD t
------------------------------
One network handles every noise level, so it must know which one it faces --
removing a whisper of noise and removing almost everything are different jobs.
``t`` is embedded (sinusoidally, as in transformers) and fed in alongside ``x``.

Sohl-Dickstein et al. (2015); Ho, Jain & Abbeel (2020).
"""
import numpy as np

from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module
from .autoencoder import _mlp


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """How fast to destroy the data.

    Too fast and the early steps are unrecoverable; too slow and T must be huge.

    The familiar constants 1e-4 and 0.02 are the original paper's, and they are
    calibrated for T=1000 -- they are a per-step noise RATE, so they only reach
    ``alpha_bar ~ 0`` if you take a thousand steps. Used unscaled at T=200 they
    leave ``alpha_bar_T ~ 0.13``: the forward process never finishes, the data is
    still visible at the end of the chain, and yet sampling starts from pure
    N(0, I). The model is then asked at generation time for a noise level it
    never saw in training, which is a silent train/sample mismatch rather than an
    error.

    So the rate is scaled by ``1000 / T``: the total noise injected over the chain
    stays what the constants intend, whatever T is.

    The clip is not cosmetic. ``beta`` is the variance of one noising step and
    ``alpha = 1 - beta`` is what survives it, so ``beta >= 1`` destroys the sample
    outright -- and the reverse step divides by ``sqrt(alpha)``, which is then a
    division by zero. Scaling alone reaches ``beta = 1`` at T=20, so the bound is
    what keeps short chains numerically valid rather than quietly producing NaN.
    """
    scale = 1000.0 / T
    return np.clip(np.linspace(beta_start * scale, beta_end * scale, T),
                   0.0, 0.999)


def cosine_beta_schedule(T, s=0.008):
    """Nichol & Dhariwal (2021): destroy the signal more gently in the middle.

    The linear schedule wipes the signal out too early -- at T=1000 it is down to
    ``alpha_bar ~ 0.08`` by the halfway point, so the entire back half of the
    chain trains on what is already almost pure noise. That is wasted capacity.

    Defining ``alpha_bar`` to follow a cosine and reading ``beta`` back off it
    holds the midpoint near 0.5 instead, spending the budget evenly across noise
    levels. Note this is defined directly in terms of ``alpha_bar``, so unlike the
    linear schedule it is parameterised by the chain's PROGRESS rather than a
    per-step rate -- it needs no rescaling when T changes.
    """
    t = np.linspace(0, T, T + 1) / T
    alpha_bar = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0, 0.999)


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the timestep, as in transformer position encoding.

    An integer input would give the network almost nothing to work with -- a
    single scalar it must expand into a notion of "how noisy". A spread of
    frequencies makes both coarse and fine differences in ``t`` linearly readable
    from the start.
    """
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / max(half - 1, 1))
    args = np.asarray(t)[:, None] * freqs[None]
    emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
    return emb if dim % 2 == 0 else np.pad(emb, ((0, 0), (0, 1)))


class NoisePredictor(Module):
    """eps_theta(x_t, t): given a noisy sample and its noise level, name the noise.

    A plain MLP over ``concat(x, timestep_embedding(t))``. Image models use a
    U-Net here; the architecture is not the idea, and the idea is what this
    module is for.
    """

    def __init__(self, n_features, hidden=(128, 128), time_dim=32, rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self.time_dim = time_dim
        self.net = _mlp([n_features + time_dim, *hidden, n_features], rng)

    def forward(self, x, t):
        x = Tensor._wrap(x)
        emb = Tensor(timestep_embedding(t, self.time_dim))
        return self.net(Tensor.concatenate([x, emb], axis=1))


class DDPM(Module):
    """Denoising diffusion probabilistic model.

    OPTIMIZATION: every schedule-derived quantity -- ``alpha_bar``, its square
    roots, the posterior coefficients -- depends only on the betas, not on the
    data. They are all precomputed once at construction into arrays indexed by
    ``t``, so a training step is a fancy-index and a broadcast rather than a
    recomputation. The whole schedule is a few kilobytes; recomputing it inside
    the loop would be the single most expensive thing in the model.
    """

    def __init__(self, n_features, T=200, hidden=(128, 128), schedule="linear",
                 rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self._rng = rng
        self.T = T
        self.n_features = n_features
        self.model = NoisePredictor(n_features, hidden, rng=rng)

        betas = (linear_beta_schedule(T) if schedule == "linear"
                 else cosine_beta_schedule(T))
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = np.cumprod(self.alphas)
        # precomputed: see the class docstring
        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1.0 - self.alpha_bars)

    def q_sample(self, x_0, t, noise=None):
        """Jump straight to step t. The closed form -- no chain is simulated."""
        x_0 = np.asarray(x_0)
        if noise is None:
            noise = self._rng.normal(size=x_0.shape)
        a = self.sqrt_alpha_bars[t][:, None]
        b = self.sqrt_one_minus_alpha_bars[t][:, None]
        return a * x_0 + b * noise, noise

    def loss(self, x_0):
        """Sample a timestep, noise the data, ask the network which noise it was."""
        x_0 = np.asarray(x_0)
        t = self._rng.randint(0, self.T, size=len(x_0))
        x_t, noise = self.q_sample(x_0, t)
        return ((self.model(x_t, t) - Tensor(noise)) ** 2).mean()

    def p_sample_step(self, x_t, t, rng):
        """One reverse step: subtract the predicted noise, then re-noise a little.

        The re-noising is not a mistake. The reverse process is STOCHASTIC -- each
        step samples from a posterior rather than taking its mean. Drop the noise
        and every trajectory from the same start collapses to the same output;
        the variety of the samples comes from exactly this term. The last step
        (t=0) is the exception: there is nothing left to sample, so it returns
        the mean.
        """
        idx = np.full(len(x_t), t)
        eps = self.model(x_t, idx).data

        mean = (x_t - self.betas[t] / self.sqrt_one_minus_alpha_bars[t] * eps) \
            / np.sqrt(self.alphas[t])
        if t == 0:
            return mean
        return mean + np.sqrt(self.betas[t]) * rng.normal(size=x_t.shape)

    def sample(self, n, rng=None, return_trajectory=False):
        """Generate: start from pure noise and walk the chain back to t=0."""
        rng = check_random_state(rng if rng is not None else self._rng)
        x = rng.normal(size=(n, self.n_features))
        traj = [x]
        was_training = self.training
        self.eval()
        for t in reversed(range(self.T)):
            x = self.p_sample_step(x, t, rng)
            if return_trajectory:
                traj.append(x)
        self.train(was_training)
        return (x, traj) if return_trajectory else x


__all__ = ["DDPM", "NoisePredictor", "linear_beta_schedule",
           "cosine_beta_schedule", "timestep_embedding"]
