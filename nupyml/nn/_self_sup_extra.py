"""Two more self-supervised methods: Barlow Twins and BYOL.

They join ``SimCLR`` and ``MaskedAutoEncoder``. Both learn representations with NO
labels from PAIRS OF AUGMENTED VIEWS, but they avoid SimCLR's dependence on many
NEGATIVE examples -- the reason SimCLR needs huge batches.
"""
import numpy as np

from ..autograd import Tensor
from .module import Module
from .layers import Linear, ReLU


def _mlp(dims, rng):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers.append(Linear(a, b, rng=rng))
        layers.append(ReLU())
    return layers[:-1]                                # drop trailing ReLU


class _Seq(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class BarlowTwins(Module):
    """Redundancy-reduction self-supervision (Zbontar et al., 2021).

    THE OBJECTIVE, WITH NO NEGATIVES
    --------------------------------
    Take two augmented views of a batch, embed both, and compute the
    CROSS-CORRELATION matrix ``C`` between the two views' (batch-normalised)
    embedding dimensions. Push ``C`` toward the IDENTITY::

        loss = sum_i (C_ii - 1)^2  +  lambda * sum_{i!=j} C_ij^2

    The diagonal term (``->1``) makes each dimension INVARIANT to the augmentation
    (agree across views); the off-diagonal term (``->0``) DECORRELATES the
    dimensions so they carry non-redundant information. Because the target is a
    fixed matrix, not other samples, it needs no negatives and no large batch --
    the elegant escape from SimCLR's batch hunger.
    """

    def __init__(self, n_features, hidden=(64,), embed_dim=32, lambda_=0.005,
                 rng=None):
        super().__init__()
        from ..utils import check_random_state
        rng = check_random_state(rng)
        self.encoder = _Seq(_mlp([n_features, *hidden, embed_dim], rng))
        self.lambda_ = lambda_
        self.embed_dim = embed_dim

    def parameters(self):
        return self.encoder.parameters()

    def encode(self, x):
        return self.encoder(Tensor._wrap(x))

    def forward(self, x):
        return self.encode(x)

    def loss(self, view_a, view_b):
        za, zb = self.encode(view_a), self.encode(view_b)
        n = za.shape[0]
        # batch-normalise each embedding dimension (mean 0, std 1 over the batch)
        za = (za - za.mean(axis=0)) / (za.var(axis=0).sqrt() + 1e-5)
        zb = (zb - zb.mean(axis=0)) / (zb.var(axis=0).sqrt() + 1e-5)
        C = (za.T @ zb) * (1.0 / n)                   # cross-correlation matrix
        eye = np.eye(self.embed_dim)
        # diagonal entries pushed to 1 (invariance), off-diagonal to 0 (decorrelation)
        on_diag = (((C * Tensor(eye)).sum(axis=1) - 1.0) ** 2).sum()
        off_diag = ((C * Tensor(1.0 - eye)) ** 2).sum()
        return on_diag + self.lambda_ * off_diag


class BYOL(Module):
    """Bootstrap Your Own Latent -- self-supervision with NO negatives at all
    (Grill et al., 2020).

    THE SURPRISE
    ------------
    SimCLR needs negatives to stop the encoder collapsing to a constant. BYOL has
    none and still does not collapse, via an ASYMMETRIC two-network trick: an
    ONLINE network (encoder + projector + an extra PREDICTOR) is trained to
    predict the projection produced by a TARGET network -- which is just an
    exponential moving average of the online weights, with the gradient STOPPED on
    its side. The predictor + stop-gradient + slowly-moving target together make
    the trivial constant solution unstable, so useful features emerge from
    agreement between views alone. ``update_target`` performs the EMA step after
    each optimiser step.
    """

    def __init__(self, n_features, hidden=(64,), embed_dim=32, proj_dim=16,
                 tau=0.99, rng=None):
        super().__init__()
        from ..utils import check_random_state
        rng = check_random_state(rng)
        self.online_encoder = _Seq(_mlp([n_features, *hidden, embed_dim], rng))
        self.online_proj = _Seq(_mlp([embed_dim, embed_dim, proj_dim], rng))
        self.predictor = _Seq(_mlp([proj_dim, proj_dim, proj_dim], rng))
        # target networks: same architecture, weights are EMA copies (no grad)
        self.target_encoder = _Seq(_mlp([n_features, *hidden, embed_dim], rng))
        self.target_proj = _Seq(_mlp([embed_dim, embed_dim, proj_dim], rng))
        self.tau = tau
        self._sync_target(1.0)                        # start target = online

    def _online_params(self):
        return self.online_encoder.parameters() + self.online_proj.parameters()

    def _target_params(self):
        return self.target_encoder.parameters() + self.target_proj.parameters()

    def parameters(self):
        return (self.online_encoder.parameters() + self.online_proj.parameters()
                + self.predictor.parameters())       # only the online path trains

    def _sync_target(self, tau):
        for tp, op in zip(self._target_params(), self._online_params()):
            tp.data[...] = tau * op.data + (1 - tau) * tp.data

    def update_target(self):
        """EMA step: nudge the target weights toward the online weights."""
        self._sync_target(1 - self.tau)

    def encode(self, x):
        return self.online_encoder(Tensor._wrap(x))

    def forward(self, x):
        return self.encode(x)

    @staticmethod
    def _norm(z):
        return z / ((z * z).sum(axis=1).sqrt().reshape(-1, 1) + 1e-8)

    def _loss_dir(self, view_online, view_target):
        p = self.predictor(self.online_proj(self.online_encoder(Tensor._wrap(view_online))))
        with_grad_target = self.target_proj(self.target_encoder(Tensor._wrap(view_target)))
        z = Tensor(with_grad_target.data)             # stop-gradient on the target
        p, z = self._norm(p), self._norm(z)
        return (2 - 2 * (p * z).sum(axis=1)).mean()   # 2 - 2*cosine similarity

    def loss(self, view_a, view_b):
        # symmetric: predict each view's target from the other view's online path
        return self._loss_dir(view_a, view_b) + self._loss_dir(view_b, view_a)


__all__ = ["BarlowTwins", "BYOL"]
