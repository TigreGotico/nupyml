"""Losses: the single number that defines what "wrong" means.

Choosing a loss is choosing what to be sensitive to:

* ``MSELoss`` squares the error, so one large mistake dominates many small
  ones. The right choice when big errors are disproportionately costly, the
  wrong one when outliers are just noise.
* ``MAELoss`` is linear, so it is robust to outliers -- but its gradient has
  constant magnitude, so it takes the same size step whether it is far away or
  nearly correct, and converges poorly near the optimum.
* ``HuberLoss`` is quadratic near zero and linear far away: MSE's clean
  gradient where you are close, MAE's robustness where you are not.
* ``CrossEntropyLoss`` for classification -- see its docstring for why not MSE.

WHY THESE ARE FUSED RATHER THAN COMPOSED
----------------------------------------
``CrossEntropyLoss`` and ``BCEWithLogitsLoss`` take LOGITS, not probabilities,
and do the squashing internally. That is not a convenience: it is the only way
to keep the computation stable. Softmax followed by log will underflow to
``log(0) = -inf`` for a confidently wrong prediction, and the gradient becomes
``nan``. Fusing the two lets the log undo the exp algebraically, so the tiny
number is never formed.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


class MSELoss(Module):
    def forward(self, pred, target):
        target = Tensor._wrap(target)
        return ((pred - target.detach()) ** 2).mean()


class MAELoss(Module):
    def forward(self, pred, target):
        target = Tensor._wrap(target)
        return (pred - target.detach()).abs().mean()


class HuberLoss(Module):
    """Quadratic for small errors, linear for large ones.

    ``delta`` is the crossover. Within it, the loss behaves like MSE, giving a
    gradient proportional to the error -- so steps shrink as you approach the
    answer, and it converges cleanly. Beyond it, the loss goes linear and the
    gradient SATURATES at a constant, so an outlier can pull only so hard no
    matter how far away it is.

    That bounded influence is the entire point, and it is the same idea as
    ``HuberRegressor`` in ``nupyml.linear_model``.
    """

    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        target = Tensor._wrap(target).detach()
        diff = pred - target
        absdiff = diff.abs()
        quad = diff * diff * 0.5
        lin = absdiff * self.delta - 0.5 * self.delta ** 2
        return Tensor.where(absdiff.data <= self.delta, quad, lin).mean()


class CrossEntropyLoss(Module):
    """Negative log-likelihood over softmax probabilities, from raw logits.

    Only the probability assigned to the TRUE class enters the loss::

        L = -mean(log p_correct)

    Nothing is said about how the remaining probability is distributed -- but
    since softmax normalises, raising the true class necessarily lowers the
    rest.

    WHY NOT SQUARED ERROR ON THE PROBABILITIES
    ------------------------------------------
    Two reasons, both decisive:

    * *The gradient vanishes exactly when it must not.* MSE through a softmax
      multiplies by the squashing function's derivative, which is nearly zero
      for a saturated output. A confidently WRONG prediction would therefore
      produce almost no gradient -- the model would be most stuck precisely
      where it is most wrong. Cross-entropy's gradient is ``p - y``, which is
      LARGEST when the prediction is worst.
    * *It punishes proportionally.* Log loss grows without bound as the true
      class's probability tends to zero. MSE caps out at 1.

    IMPLEMENTATION
    --------------
    Fused ``log_softmax`` plus a lookup. The full one-hot matrix product is
    unnecessary -- multiplying by zeros for every wrong class -- so the correct
    class's log-probability is simply indexed out.
    """

    def forward(self, logits, target):
        target = np.asarray(target if not isinstance(target, Tensor) else target.data,
                            dtype=np.int64)
        logp = F.log_softmax(logits, axis=-1)
        n = logp.shape[0]
        picked = logp[np.arange(n), target]
        return -picked.mean()


class NLLLoss(Module):
    """Negative log likelihood over log-probabilities."""

    def forward(self, logp, target):
        target = np.asarray(target if not isinstance(target, Tensor) else target.data,
                            dtype=np.int64)
        n = logp.shape[0]
        return -logp[np.arange(n), target].mean()


class BCELoss(Module):
    """Binary cross-entropy over probabilities in (0, 1)."""

    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        target = Tensor._wrap(target).detach()
        p = pred.clip(self.eps, 1 - self.eps)
        return -(target * p.log() + (1.0 - target) * (1.0 - p).log()).mean()


class BCEWithLogitsLoss(Module):
    """Binary cross-entropy from logits, computed the stable way.

    ``BCELoss(sigmoid(x))`` is the same function mathematically and unusable in
    practice: ``sigmoid`` saturates to exactly 0.0 or 1.0 in float, and ``log``
    of that is ``-inf``.

    Algebra removes the problem entirely::

        L = max(x, 0) - x*y + log(1 + exp(-|x|))

    Every term is bounded: ``exp`` only ever sees a NEGATIVE argument, so it
    cannot overflow, and ``log1p``-style behaviour keeps the small case
    accurate. This identity is worth recognising -- the same rearrangement
    appears wherever a log meets a sigmoid.

    Use this over ``BCELoss`` unless the input genuinely is a probability from
    somewhere else.
    """

    def forward(self, logits, target):
        target = Tensor._wrap(target).detach()
        relu_x = logits.relu()
        softplus = ((-logits.abs()).exp() + 1.0).log()
        return (relu_x - logits * target + softplus).mean()


__all__ = ["MSELoss", "MAELoss", "HuberLoss", "CrossEntropyLoss", "NLLLoss",
           "BCELoss", "BCEWithLogitsLoss"]
