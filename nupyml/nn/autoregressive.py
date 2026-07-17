"""Autoregressive models: an exact likelihood from the chain rule alone.

THE IDEA
--------
Every joint distribution factorises. That is not an assumption or an
approximation -- it is the chain rule of probability, and it is always true::

    p(x) = p(x_1) * p(x_2 | x_1) * p(x_3 | x_1, x_2) * ...

So modelling a joint distribution over d variables reduces to d ordinary
conditional predictions. Each one is a normalized distribution over a single
variable, which is something we already know how to do. Multiply them (add the
logs) and you have ``log p(x)`` exactly -- no bound like the VAE's, and no
invertibility constraint like a flow's.

This is the same idea that underlies language models. Predicting the next token
given the previous ones IS this factorisation; nothing more sophisticated is
happening.

THE PRICE
---------
Sampling is SEQUENTIAL. You cannot draw ``x_3`` until you know ``x_2``, so
generation costs d forward passes and cannot be parallelised. Training, by
contrast, is one pass -- the true values are already known, so every conditional
can be evaluated at once. Fast training, slow sampling: the exact inverse of a
GAN, and the reason autoregressive image models lost to diffusion at scale while
autoregressive text models did not.

    VAE:   bound on likelihood, fast sampling
    flow:  exact likelihood, fast sampling, invertibility constraint
    AR:    exact likelihood, no constraint, sequential sampling
    GAN:   no likelihood, fast sampling

THE PROBLEM: HOW DO YOU DO IT IN ONE PASS?
------------------------------------------
The naive way is d separate networks, one per conditional. That is wasteful and
does not scale.

MADE's answer: take ONE autoencoder and MASK its weights so that output ``i``
can only ever see inputs ``< i``. The network computes all d conditionals
simultaneously, sharing all its features, and the masking guarantees the
autoregressive property structurally -- not by convention, and not by anything
the training procedure has to be careful about.

If that guarantee were violated the model would be free to look at ``x_i`` when
predicting ``x_i``, and would learn the identity: perfect training likelihood,
worthless samples. It would look like a triumph. This is why the masks matter,
and why the test for this module checks the dependency structure directly rather
than trusting the loss.

Germain, Gregor, Larochelle & Murray (2015).
"""
import numpy as np

from ..autograd import Tensor
from ..utils import check_random_state
from .module import Module, ModuleList, Parameter
from .layers import Linear


class MaskedLinear(Linear):
    """A Linear layer with a fixed binary mask applied to its weights.

    The mask multiplies the weight matrix on every forward pass. Masked entries
    contribute nothing to the output, so they also receive no gradient -- the
    connection is not merely discouraged, it does not exist.

    The mask is shaped like the weight, ``(in_features, out_features)``, since
    the layer computes ``x @ W``.
    """

    def __init__(self, in_features, out_features, rng=None):
        super().__init__(in_features, out_features, rng=rng)
        self.mask = np.ones((in_features, out_features))

    def set_mask(self, mask):
        self.mask = np.asarray(mask, dtype=np.float64)

    def forward(self, x):
        out = Tensor._wrap(x) @ (self.weight * Tensor(self.mask))
        return out + self.bias if self.bias is not None else out


class MADE(Module):
    """Masked autoencoder for distribution estimation, over binary data.

    HOW THE MASKS ARE BUILT
    -----------------------
    Give every unit a number ``m``:

    * input ``i`` gets ``m = i`` -- its position in the ordering;
    * hidden units get ``m`` drawn from ``1 .. d-1``;
    * output ``i`` gets ``m = i`` again.

    Then connect unit ``a`` to unit ``b`` only when the information can legally
    flow:

    * hidden layers: ``m_hidden >= m_input`` -- a hidden unit may depend on any
      input at or below its number;
    * output layer: ``m_output > m_hidden`` -- STRICTLY greater, and this is the
      whole trick. Output ``i`` connects only to hidden units numbered ``< i``,
      each of which saw only inputs ``<= its own number < i``. So output ``i``
      can depend on ``x_1 .. x_{i-1}`` and provably not on ``x_i``.

    The strict inequality is the difference between a generative model and an
    identity function. Everything else is bookkeeping.

    ``forward`` returns LOGITS for each dimension -- the parameters of d Bernoulli
    conditionals, all computed in a single pass.
    """

    def __init__(self, n_features, hidden=(64, 64), rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self._rng = rng
        self.n_features = n_features

        sizes = [n_features, *hidden, n_features]
        self.layers = ModuleList([MaskedLinear(sizes[i], sizes[i + 1], rng=rng)
                                  for i in range(len(sizes) - 1)])
        self._build_masks(hidden, rng)

    def _build_masks(self, hidden, rng):
        d = self.n_features
        # the input ordering; degrees 0..d-1
        degrees = [np.arange(d)]
        for h in hidden:
            # a hidden unit numbered d-1 could depend on every input, which is
            # useless for the last output; numbering from 0..d-2 keeps every
            # hidden unit reachable by at least one output
            degrees.append(rng.randint(degrees[-1].min(), d - 1, size=h))
        degrees.append(np.arange(d))

        for i, layer in enumerate(self.layers):
            prev, cur = degrees[i], degrees[i + 1]
            # shaped (in, out) to match the weight -- see MaskedLinear.
            # the last layer is the strict one, which is what forbids output i
            # from seeing input i; see the class docstring
            mask = (cur[None, :] > prev[:, None]) if i == len(self.layers) - 1 \
                else (cur[None, :] >= prev[:, None])
            layer.set_mask(mask.astype(np.float64))
        self.degrees = degrees

    def forward(self, x):
        h = Tensor._wrap(x)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = h.relu()
        return h

    def log_prob(self, x):
        """Sum the log-conditionals: the chain rule, written out.

        Uses the numerically stable ``logsigmoid`` forms rather than taking the
        log of a sigmoid, which underflows to -inf for confident predictions.
        """
        x = np.asarray(x, dtype=np.float64)
        logits = self(x)
        t = Tensor(x)
        # log p = x*log(sigmoid(z)) + (1-x)*log(1-sigmoid(z)), stably:
        #   log(sigmoid(z))   = -softplus(-z)
        #   log(1-sigmoid(z)) = -softplus(z)
        return -(t * _softplus(-logits) + (1.0 - t) * _softplus(logits)).sum(axis=-1)

    def loss(self, x):
        return -self.log_prob(x).mean()

    def sample(self, n, rng=None):
        """Sequential generation -- d passes, one per dimension.

        The loop cannot be vectorized over dimensions, and that is not an
        implementation shortcoming: dimension ``i`` genuinely does not exist
        until ``i-1`` has been drawn. It is the defining cost of the family.
        """
        rng = check_random_state(rng if rng is not None else self._rng)
        x = np.zeros((n, self.n_features))
        was_training = self.training
        self.eval()
        for i in range(self.n_features):
            p = _sigmoid_np(self(x).data[:, i])
            x[:, i] = (rng.uniform(size=n) < p).astype(np.float64)
        self.train(was_training)
        return x


def _softplus(t):
    """log(1 + exp(t)) on the tape, without overflowing.

    ``exp(t)`` overflows for t around 700, so the identity
    ``softplus(t) = max(t, 0) + log(1 + exp(-|t|))`` is used instead: the
    exponent is never positive, and the ``max(t, 0)`` term carries the linear
    growth exactly.
    """
    return t.relu() + ((-t.abs()).exp() + 1.0).log()


def _sigmoid_np(x):
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


__all__ = ["MADE", "MaskedLinear"]
