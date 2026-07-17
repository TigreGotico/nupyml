"""Normalizing flows: exact likelihood by never losing information.

THE IDEA
--------
A VAE and a GAN both give up on the exact likelihood -- one bounds it, the other
ignores it. A flow refuses to. It builds a distribution by pushing a simple one
(a gaussian) through an INVERTIBLE map::

    z ~ N(0, I)  ------ f ----->  x = f(z)          (generate)
    x            <----- f^-1 ---  z = f^-1(x)       (infer, exactly)

Because ``f`` is a bijection, no information is destroyed and the change-of-
variables formula gives the density exactly::

    log p(x) = log p_z(f^-1(x)) + log |det J_{f^-1}(x)|

Read that second term carefully -- it is the entire subject. Squeezing a region
of space concentrates probability there; the Jacobian determinant measures that
volume change and corrects for it. Without it you could "cheat" by shrinking
everything toward a high-density point.

So a flow gives you what neither of its rivals can: ``log p(x)`` for any x,
exactly, with no bound and no sampling.

THE COST
--------
``f`` must be invertible AND its Jacobian determinant must be cheap. In general a
determinant is O(d^3), which would make every training step hopeless. The whole
literature is a hunt for architectures where it is not.

THE COUPLING TRICK (RealNVP)
----------------------------
Split the input in half. Copy one half through untouched, and use it to compute a
scale and shift for the other::

    y_a = x_a                                   (identity)
    y_b = x_b * exp(s(x_a)) + t(x_a)            (affine, conditioned on x_a)

Two properties fall out, and both are the point:

* **Trivially invertible**, without ever inverting ``s`` or ``t``: given ``y``,
  you have ``y_a = x_a`` for free, so recompute ``s(x_a)``, ``t(x_a)`` and undo
  the affine map. The networks can be arbitrarily complex -- they are never
  inverted, only re-evaluated.
* **Triangular Jacobian.** ``y_a`` does not depend on ``x_b`` at all, so the
  Jacobian is block-triangular and its determinant is just the product of the
  diagonal -- ``sum(s(x_a))``. An O(d^3) determinant collapses to a sum.

The identity half is the price: it is unchanged by this layer. So layers alternate
which half they freeze, and stacking a few gives every dimension its turn.

Dinh, Sohl-Dickstein & Bengio (2016). Glow adds 1x1 convolutions to learn the
permutation rather than alternating it.
"""
import numpy as np

from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, ModuleList, Parameter
from .autoencoder import _mlp


class AffineCoupling(Module):
    """One RealNVP layer: copy half, affine-transform the other half.

    ``mask`` selects which coordinates pass through untouched (1) and which are
    transformed (0). Alternate it between layers or the frozen half never learns.

    The scale network's output is passed through ``tanh`` before exponentiating.
    Unbounded ``s`` would make ``exp(s)`` overflow the moment training wobbles,
    and the log-determinant with it; bounding the log-scale to [-1, 1] costs
    expressiveness that stacking gets back.
    """

    def __init__(self, dim, mask, hidden=(64,), rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self.mask = np.asarray(mask, dtype=np.float64)
        self.scale_net = _mlp([dim, *hidden, dim], rng)
        self.shift_net = _mlp([dim, *hidden, dim], rng)

    def _params(self, x_frozen):
        # tanh-bounded log-scale: see the class docstring
        s = self.scale_net(x_frozen).tanh() * (1.0 - Tensor(self.mask))
        t = self.shift_net(x_frozen) * (1.0 - Tensor(self.mask))
        return s, t

    def forward(self, x):
        """x -> y, plus log|det J|. The generative direction."""
        x = Tensor._wrap(x)
        frozen = x * Tensor(self.mask)
        s, t = self._params(frozen)
        y = frozen + (x * (1.0 - Tensor(self.mask))) * s.exp() + t
        return y, s.sum(axis=-1)

    def inverse(self, y):
        """y -> x. Note that ``s`` and ``t`` are RECOMPUTED, never inverted."""
        y = Tensor._wrap(y)
        frozen = y * Tensor(self.mask)
        s, t = self._params(frozen)
        x = frozen + ((y - t) * (1.0 - Tensor(self.mask))) * (-s).exp()
        return x, -s.sum(axis=-1)


class RealNVP(Module):
    """A stack of coupling layers, with the masks alternating.

    The log-determinants simply ADD across layers -- a composition of maps has the
    product of Jacobians, so the log-dets sum. That is why depth is free here:
    each layer contributes one more cheap term, never an expensive determinant.

    Trained by maximum likelihood directly: ``loss = -mean(log_prob(x))``. No
    bound, no adversary, no sampling. Among generative models that is rare enough
    to be the reason flows exist.
    """

    def __init__(self, dim, n_layers=6, hidden=(64,), rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self.dim = dim
        # alternate which half is frozen so every coordinate gets transformed
        base = np.arange(dim) % 2
        self.layers = ModuleList([
            AffineCoupling(dim, base if i % 2 == 0 else 1 - base, hidden, rng)
            for i in range(n_layers)])

    def forward(self, z):
        """Generate: prior sample -> data."""
        log_det = Tensor(np.zeros(Tensor._wrap(z).shape[0]))
        x = Tensor._wrap(z)
        for layer in self.layers:
            x, ld = layer(x)
            log_det = log_det + ld
        return x, log_det

    def inverse(self, x):
        """Infer: data -> prior sample. Layers unwind in reverse."""
        log_det = Tensor(np.zeros(Tensor._wrap(x).shape[0]))
        z = Tensor._wrap(x)
        for layer in reversed(list(self.layers)):
            z, ld = layer.inverse(z)
            log_det = log_det + ld
        return z, log_det

    def log_prob(self, x):
        """The exact log-density -- change of variables, term for term."""
        z, log_det = self.inverse(x)
        log_pz = ((z ** 2) * -0.5 - 0.5 * np.log(2 * np.pi)).sum(axis=-1)
        return log_pz + log_det

    def loss(self, x):
        return -self.log_prob(x).mean()

    def sample(self, n, rng=None):
        rng = check_random_state(rng)
        z = Tensor(rng.normal(size=(n, self.dim)))
        return self(z)[0].data


__all__ = ["AffineCoupling", "RealNVP"]
