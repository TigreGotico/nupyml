"""Variational graph autoencoder: embed a graph by reconstructing its edges."""
import numpy as np

from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, Parameter
from .layers import Linear


def _mlp(dims, rng):
    return [Linear(dims[i], dims[i + 1], rng=rng) for i in range(len(dims) - 1)]


def _run(layers, x, act=True):
    h = x
    for i, l in enumerate(layers):
        h = l(h)
        if act and i < len(layers) - 1:
            h = h.relu()
    return h


class VGAE(Module):
    """Embed a graph by RECONSTRUCTING its edges, probabilistically (Kipf & Welling).

    An autoencoder for graphs: a GCN ENCODER maps each node to a Gaussian latent
    (mean and variance), and the DECODER is just the inner product -- the probability
    of an edge between two nodes is ``sigmoid(z_i . z_j)``. Training reconstructs the
    adjacency while a KL term regularises the latents toward a standard normal, so the
    embedding is smooth and generative. The learned ``z`` captures community and
    proximity structure and is a strong link-prediction feature. Two-layer GCN encoder.
    """

    def __init__(self, in_dim, hidden=32, latent=16, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.gc1 = Linear(in_dim, hidden, rng=r)
        self.gc_mu = Linear(hidden, latent, rng=r)
        self.gc_logvar = Linear(hidden, latent, rng=r)
        self.latent = latent
        self._rng = r

    def parameters(self):
        return (list(self.gc1.parameters()) + list(self.gc_mu.parameters())
                + list(self.gc_logvar.parameters()))

    @staticmethod
    def _norm_adj(A):
        A = np.asarray(A, float) + np.eye(len(A))          # add self-loops
        d = A.sum(axis=1)
        dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        return dinv[:, None] * A * dinv[None, :]

    def encode(self, X, A):
        An = Tensor(self._norm_adj(A))
        h = (An @ self.gc1(Tensor._wrap(X))).relu()        # GCN layer
        mu = An @ self.gc_mu(h)
        logvar = An @ self.gc_logvar(h)
        return mu, logvar

    def forward(self, X, A):
        mu, logvar = self.encode(X, A)
        eps = Tensor(self._rng.randn(*mu.shape))
        z = mu + eps * (logvar * 0.5).exp()                # reparameterisation
        return z, mu, logvar

    def loss(self, X, A):
        z, mu, logvar = self.forward(X, A)
        logits = z @ z.transpose()                         # inner-product decoder
        At = Tensor(np.asarray(A, float))
        # edge reconstruction (BCE with logits) + KL to N(0, I)
        recon = ((1.0 + logits.exp()).log() - At * logits).mean()
        kl = (-0.5 * (1.0 + logvar - mu * mu - logvar.exp()).mean())
        return recon + kl * (1.0 / len(np.asarray(A)))

    def embed(self, X, A):
        return self.encode(X, A)[0].data


__all__ = ["ConditionalNeuralProcess", "PointNet", "EnergyBasedModel", "SimSiam",
           "VGAE"]


__all__ = ["VGAE"]
