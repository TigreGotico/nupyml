"""Neural architectures v7: amortised regression, point clouds, energy models,
non-contrastive self-supervision, and graph autoencoders.

Five more architectures. The Conditional Neural Process learns to regress a whole
FAMILY of functions with uncertainty. PointNet is permutation-invariant over an
unordered POINT SET. The energy-based model learns an unnormalised density and
samples it with Langevin dynamics. SimSiam learns representations from augmentations
with NO negatives. The variational graph autoencoder embeds a graph by reconstructing
its edges.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
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


class ConditionalNeuralProcess(Module):
    """Learn to regress a whole FAMILY of functions, with uncertainty
    (Garnelo et al., 2018).

    A Gaussian process regresses one function beautifully but costs ``O(n^3)`` and
    must be re-fit per dataset. A Conditional Neural Process AMORTISES that: an encoder
    maps each observed (x, y) CONTEXT point to a vector, averages them into a single
    representation of "which function is this", and a decoder predicts the mean AND
    variance at any target x from that representation. Trained across many functions,
    it produces GP-like predictions -- uncertainty that shrinks where context is dense
    -- in a single forward pass. ``forward`` takes context (x, y) and target x.
    """

    def __init__(self, x_dim=1, y_dim=1, hidden=32, r_dim=32, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.encoder = _mlp([x_dim + y_dim, hidden, r_dim], r)
        self.decoder = _mlp([r_dim + x_dim, hidden, 2 * y_dim], r)
        self.y_dim = y_dim

    def parameters(self):
        return [p for l in (self.encoder + self.decoder) for p in l.parameters()]

    def forward(self, cx, cy, tx):
        cx = Tensor._wrap(cx); cy = Tensor._wrap(cy); tx = Tensor._wrap(tx)
        r = _run(self.encoder, Tensor.concatenate([cx, cy], axis=1))
        r = r.mean(axis=0, keepdims=True)                 # aggregate the context
        r_rep = Tensor(np.tile(r.data, (tx.shape[0], 1)))
        out = _run(self.decoder, Tensor.concatenate([r_rep, tx], axis=1))
        mean = out[:, :self.y_dim]
        log_sigma = out[:, self.y_dim:]
        sigma = (0.1 + (log_sigma * 0.5).exp())           # positive std
        return mean, sigma

    def loss(self, cx, cy, tx, ty):
        mean, sigma = self.forward(cx, cy, tx)
        ty = Tensor._wrap(ty)
        return (0.5 * (((ty - mean) / sigma) ** 2) + sigma.log()).mean()


class PointNet(Module):
    """A network invariant to the ORDER of its inputs (Qi et al., 2017).

    A point cloud has no canonical ordering -- the same shape can be listed in any
    permutation -- so a network reading it must give the SAME answer regardless of
    order. PointNet does this with a shared per-point MLP followed by a SYMMETRIC
    aggregation (max-pooling over points): each point is embedded independently, then
    a permutation-invariant pool collapses them to one global descriptor that a
    classifier reads. Simple, and it was the breakthrough that let deep nets consume
    raw 3-D points. Input ``(batch, n_points, point_dim)``.
    """

    def __init__(self, point_dim=3, n_classes=2, hidden=64, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.point_mlp = _mlp([point_dim, hidden, hidden], r)
        self.head = _mlp([hidden, hidden, n_classes], r)

    def parameters(self):
        return [p for l in (self.point_mlp + self.head) for p in l.parameters()]

    def forward(self, clouds):
        x = Tensor._wrap(clouds)
        B, N, D = x.shape
        feats = _run(self.point_mlp, x.reshape(B * N, D)).reshape(B, N, -1)
        pooled = feats.max(axis=1)                         # symmetric pool -> invariant
        return _run(self.head, pooled)


class EnergyBasedModel(Module):
    """Learn an unnormalised density and SAMPLE it with Langevin (LeCun; Du & Mordatch).

    A normalising flow must stay invertible; an autoregressive model must factorise.
    An energy-based model is free of both: it learns a scalar ENERGY ``E(x)`` that is
    LOW on data and high elsewhere -- the density is ``exp(-E)/Z`` with the intractable
    ``Z`` never computed. Training uses contrastive divergence: push DOWN the energy of
    real data and UP the energy of "negative" samples drawn by LANGEVIN dynamics
    (gradient descent on energy plus noise). Sampling is the same Langevin walk. The
    most flexible generative model, at the cost of MCMC in the loop.
    """

    def __init__(self, dim, hidden=64, langevin_steps=20, langevin_lr=0.1, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.net = _mlp([dim, hidden, hidden, 1], r)
        self.dim = dim
        self.langevin_steps = langevin_steps
        self.langevin_lr = langevin_lr

    def parameters(self):
        return [p for l in self.net for p in l.parameters()]

    def energy(self, x):
        return _run(self.net, Tensor._wrap(x)).reshape(-1)

    def sample(self, n, rng=None, init=None):
        rng = check_random_state(rng)
        x = rng.randn(n, self.dim) if init is None else np.array(init, float)
        for _ in range(self.langevin_steps):
            xt = Tensor(x, requires_grad=True)
            e = self.energy(xt).sum()
            e.backward()
            x = x - 0.5 * self.langevin_lr * xt.grad \
                + np.sqrt(self.langevin_lr) * rng.randn(*x.shape)   # Langevin step
            x = np.clip(x, -5, 5)
        return x

    def loss(self, x_data, reg=1.0, rng=None):
        x_data = Tensor._wrap(x_data)
        rng = check_random_state(rng)
        # contrastive divergence: start the Langevin chain AT the data (+ noise) --
        # far more stable than sampling from random noise
        init = x_data.data + 0.1 * rng.randn(*x_data.data.shape)
        neg = self.sample(len(x_data.data), rng=rng, init=init)
        e_pos = self.energy(x_data)
        e_neg = self.energy(neg)
        return (e_pos.mean() - e_neg.mean()
                + reg * ((e_pos ** 2).mean() + (e_neg ** 2).mean()))


class SimSiam(Module):
    """Self-supervised learning with NO negatives, no collapse (Chen & He, 2021).

    Contrastive methods need many negative pairs to stop the network mapping
    everything to one point. SimSiam shows you can drop the negatives entirely: it
    feeds two AUGMENTED views through the same encoder and a projector, then a small
    PREDICTOR on one side must match the other side's (STOP-GRADIENT) projection.
    The stop-gradient is the whole trick -- it stops the trivial collapse the missing
    negatives would otherwise allow -- and the network learns useful invariant
    representations. Encoder + projector + predictor MLPs here.
    """

    def __init__(self, in_dim, hidden=64, proj_dim=32, rng=None):
        super().__init__()
        r = check_random_state(rng)
        self.encoder = _mlp([in_dim, hidden, hidden], r)
        self.projector = _mlp([hidden, proj_dim], r)
        self.predictor = _mlp([proj_dim, proj_dim // 2, proj_dim], r)

    def parameters(self):
        return [p for l in (self.encoder + self.projector + self.predictor)
                for p in l.parameters()]

    def encode(self, x):
        return _run(self.encoder, Tensor._wrap(x))

    def _proj(self, x):
        return _run(self.projector, self.encode(x))

    def _neg_cos(self, p, z):
        p = p / ((p * p).sum(axis=1, keepdims=True) ** 0.5 + 1e-8)
        z = z / ((z * z).sum(axis=1, keepdims=True) ** 0.5 + 1e-8)
        return -(p * z).sum(axis=1).mean()

    def loss(self, view1, view2):
        z1, z2 = self._proj(view1), self._proj(view2)
        p1 = _run(self.predictor, z1); p2 = _run(self.predictor, z2)
        # symmetric loss with stop-gradient on the target branch
        return 0.5 * (self._neg_cos(p1, z2.detach()) + self._neg_cos(p2, z1.detach()))


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
