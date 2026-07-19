"""Predict a DISTRIBUTION over the target, not a point (Bishop, 1994)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module
from .layers import Linear, LayerNorm, Embedding, Dropout, ReLU, GELU


class MixtureDensityNetwork(Module):
    """Predict a DISTRIBUTION over the target, not a point (Bishop, 1994).

    THE PROBLEM WITH POINT REGRESSION
    ---------------------------------
    Squared-error regression predicts the conditional MEAN -- which is disastrous
    when the target is MULTIMODAL (given x, y could be one of two valid values):
    the mean sits between the modes, at a value that never actually occurs (an
    inverse-kinematics arm pointing nowhere). An MDN outputs the parameters of a
    MIXTURE of Gaussians -- weights, means, and variances -- so it can say "y is
    near A OR near B", and it is trained by maximising the mixture likelihood.

    ``forward`` returns ``(pi, mu, sigma)``; ``nll`` is the training loss;
    ``predict`` returns the most-likely component's mean.
    """

    def __init__(self, in_dim, n_components=3, hidden=32, rng=None):
        super().__init__()
        self.k = n_components
        self.fc = Linear(in_dim, hidden, rng=rng)
        self.act = ReLU()
        self.pi = Linear(hidden, n_components, rng=rng)
        self.mu = Linear(hidden, n_components, rng=rng)
        self.logsigma = Linear(hidden, n_components, rng=rng)

    def parameters(self):
        return (list(self.fc.parameters()) + list(self.pi.parameters())
                + list(self.mu.parameters()) + list(self.logsigma.parameters()))

    def forward(self, x):
        h = self.act(self.fc(Tensor._wrap(x) if not isinstance(x, Tensor) else x))
        pi = F.softmax(self.pi(h), axis=-1)
        mu = self.mu(h)
        sigma = self.logsigma(h).exp() + 1e-3
        return pi, mu, sigma

    def nll(self, x, y):
        pi, mu, sigma = self.forward(x)
        y = np.asarray(y, float).reshape(-1, 1)
        # log N(y | mu_k, sigma_k) for every component
        z = (Tensor(y) - mu) / sigma
        log_comp = (-0.5 * z * z - sigma.log()
                    - 0.5 * np.log(2 * np.pi) + (pi + 1e-12).log())
        # stable log-sum-exp over components (subtract the row max as a constant)
        m = log_comp.data.max(axis=1, keepdims=True)
        lse = (log_comp - Tensor(m)).exp().sum(axis=1).log() + Tensor(m.ravel())
        return (-lse).mean()

    def predict(self, x):
        pi, mu, sigma = self.forward(x)
        best = pi.data.argmax(axis=1)                 # the dominant component's mean
        return mu.data[np.arange(len(best)), best]


__all__ = ["MixtureDensityNetwork"]
