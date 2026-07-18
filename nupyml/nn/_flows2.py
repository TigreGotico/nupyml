"""Masked Autoregressive Flow (MAF) -- an exact-likelihood density model.

RealNVP (in ``flows``) uses coupling layers; MAF uses AUTOREGRESSIVE transforms,
which are more flexible per layer. Each transform standardises ``x_i`` using a
shift and scale that depend only on ``x_{<i}`` (computed by a MADE-masked net), so
the Jacobian is triangular and its log-determinant is a cheap sum -- giving an
EXACT log-likelihood.
"""
import numpy as np

from ..autograd import Tensor
from .module import Module, Parameter


def _made_masks(d, hidden, rng):
    """MADE degree masks enforcing the autoregressive property (Germain, 2015)."""
    degrees = [np.arange(1, d + 1)]                    # input degrees 1..d
    for h in hidden:
        degrees.append(rng.randint(1, d, size=h))      # hidden degrees in [1, d-1]
    masks = []
    for i in range(len(degrees) - 1):
        masks.append((degrees[i + 1][:, None] >= degrees[i][None, :]).astype(float))
    # output mask: dimension i's params may depend only on inputs with degree < i
    out_mask = (degrees[0][:, None] > degrees[-1][None, :]).astype(float)
    return masks, out_mask, degrees[0]


class _MADEConditioner(Module):
    """Outputs (mu, log_scale) per dimension, autoregressively (masked MLP)."""

    def __init__(self, d, hidden, rng):
        super().__init__()
        self.d = d
        masks, out_mask, _ = _made_masks(d, hidden, rng)
        self.Ws, self.bs, self.masks = [], [], masks
        sizes = [d] + list(hidden)
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.Ws.append(Parameter(rng.randn(a, b) * 0.1))
            self.bs.append(Parameter(np.zeros(b)))
        # two heads (mu, log_scale), each masked by the output mask
        self.W_mu = Parameter(rng.randn(sizes[-1], d) * 0.1)
        self.W_s = Parameter(rng.randn(sizes[-1], d) * 0.1)
        self.b_mu = Parameter(np.zeros(d))
        self.b_s = Parameter(np.zeros(d))
        self.out_mask = out_mask.T                     # (hidden_last, d)

    def parameters(self):
        return (self.Ws + self.bs
                + [self.W_mu, self.W_s, self.b_mu, self.b_s])

    def forward(self, x):
        h = x
        for W, b, m in zip(self.Ws, self.bs, self.masks):
            h = (h @ (W * Tensor(m.T)) + b).relu()
        mu = h @ (self.W_mu * Tensor(self.out_mask)) + self.b_mu
        log_s = h @ (self.W_s * Tensor(self.out_mask)) + self.b_s
        return mu, log_s.clip(-5.0, 5.0)               # bound the scale for stability


class MAF(Module):
    """Masked Autoregressive Flow: stack autoregressive transforms (Papamakarios, 2017).

    Each transform maps data to noise by ``z_i = (x_i - mu_i(x_{<i})) *
    exp(-s_i(x_{<i}))`` with ``mu, s`` from a MADE-masked network. Because ``z_i``
    depends only on ``x_{<=i}``, the Jacobian is triangular and
    ``log|det| = -sum_i s_i`` -- so the change-of-variables formula gives the EXACT
    log-density ``log p(x) = log N(z) - sum s``. Stacking transforms (with the
    variable order reversed between them, so every dimension gets to condition on
    every other) makes it a very expressive density model. ``log_prob`` scores,
    ``fit`` maximises it.
    """

    def __init__(self, n_features, n_transforms=5, hidden=(32, 32), lr=0.01,
                 n_epochs=200, random_state=None):
        super().__init__()
        from ..utils import check_random_state
        rng = check_random_state(random_state)
        self.d = n_features
        self.transforms = [_MADEConditioner(n_features, hidden, rng)
                           for _ in range(n_transforms)]
        self.lr = lr
        self.n_epochs = n_epochs
        self._rng = rng

    def parameters(self):
        return [p for t in self.transforms for p in t.parameters()]

    def log_prob(self, X):
        x = Tensor._wrap(X)
        log_det = Tensor(np.zeros(x.shape[0]))
        for k, t in enumerate(self.transforms):
            mu, log_s = t(x)
            z = (x - mu) * (-log_s).exp()
            log_det = log_det + (-log_s).sum(axis=1)
            x = z
            if k < len(self.transforms) - 1:
                x = x[:, ::-1]                         # reverse order between flows
        # standard-normal base density on the final z
        base = (-0.5 * (x * x).sum(axis=1)
                - 0.5 * self.d * np.log(2 * np.pi))
        return base + log_det

    def fit(self, X):
        from .optim import Adam
        X = np.asarray(X, float)
        opt = Adam(self.parameters(), lr=self.lr)
        for _ in range(self.n_epochs):
            opt.zero_grad()
            loss = -self.log_prob(X).mean()            # negative log-likelihood
            loss.backward()
            opt.step()
        return self

    def score_samples(self, X):
        return self.log_prob(X).data


__all__ = ["MAF"]
