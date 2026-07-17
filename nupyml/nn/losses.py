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



# ---------------------------------------------------------------------------
# classification: reweighting what the loss pays attention to
# ---------------------------------------------------------------------------

class FocalLoss(Module):
    """Cross-entropy that stops easy examples from drowning out hard ones.

    THE PROBLEM
    -----------
    In a detector with 1000 background patches per object, cross-entropy's
    gradient is dominated by background. Each background patch is individually
    easy -- the model already assigns it p=0.99 -- and contributes only a small
    loss, but there are so many that their sum buries the handful of examples
    that still have something to teach. Reweighting by class frequency does not
    fix this: the issue is easy-versus-hard, not rare-versus-common.

    THE FIX
    -------
    Multiply each sample's loss by ``(1 - p_true) ** gamma``::

        FL = -alpha * (1 - p) ** gamma * log(p)

    For a well-classified sample (p = 0.99, gamma = 2) the factor is 0.0001 --
    its contribution essentially vanishes. For a struggling one (p = 0.1) the
    factor is 0.81, nearly untouched. The loss automatically concentrates on
    what is not yet learned, and it does so smoothly rather than by a threshold.

    ``gamma`` sets how sharply: 0 is plain cross-entropy, 2 is the usual choice.
    ``alpha`` optionally adds ordinary class weighting on top.

    Lin et al. (2017), "Focal Loss for Dense Object Detection".
    """

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        target = np.asarray(target if not isinstance(target, Tensor)
                            else target.data, dtype=np.int64)
        logp = F.log_softmax(logits, axis=-1)
        n, k = logp.shape
        logp_true = logp[np.arange(n), target]
        p_true = logp_true.exp()
        focal = (1.0 - p_true) ** self.gamma
        loss = -(focal * logp_true)
        if self.alpha is not None:
            alpha = np.asarray(self.alpha, dtype=np.float64)
            if alpha.ndim != 1 or len(alpha) != k:
                raise ValueError(
                    f"alpha must give one weight per class: expected {k}, "
                    f"got {alpha.shape}")
            loss = loss * Tensor(alpha[target])
        return loss.mean()


class LabelSmoothingCrossEntropy(Module):
    """Cross-entropy against softened targets rather than one-hot ones.

    A one-hot target asks for ``p_true = 1`` exactly, which softmax can only
    approach by driving the true logit to infinity. So the model is pushed to be
    ever more confident forever, which overfits and produces the wildly
    overconfident probabilities that calibration then has to undo.

    Label smoothing asks for ``1 - eps`` instead, spreading ``eps`` over the
    other classes::

        L = (1 - eps) * CE(true) + (eps / K) * sum_k CE(k)

    Now the optimum is a FINITE logit gap, so the model stops at a sensible
    confidence. It regularises, calibrates, and costs one line. Standard in
    transformer and image-classification training, typically eps = 0.1.
    """

    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        target = np.asarray(target if not isinstance(target, Tensor)
                            else target.data, dtype=np.int64)
        logp = F.log_softmax(logits, axis=-1)
        n, k = logp.shape
        nll = -logp[np.arange(n), target].mean()
        smooth = -logp.mean()          # mean over every class and sample
        return nll * (1.0 - self.eps) + smooth * self.eps


class KLDivLoss(Module):
    """KL divergence between a target distribution and a predicted one.

    ``KL(q || p) = sum_k q_k * (log q_k - log p_k)`` -- how many extra bits it
    costs to encode samples from q using a code built for p. Asymmetric on
    purpose: it is not a distance.

    Used wherever the target is a DISTRIBUTION rather than a label -- notably
    distillation, where the target is a teacher's soft output. Note that against
    a one-hot target KL and cross-entropy differ only by a constant, so they
    give identical gradients; KL earns its place only when q is soft.

    Takes log-probabilities as input for the same stability reason as
    ``CrossEntropyLoss``.
    """

    def forward(self, log_pred, target_prob):
        q = Tensor._wrap(target_prob).detach()
        log_q = Tensor(np.log(np.clip(q.data, 1e-12, None)))
        return (q * (log_q - log_pred)).sum(axis=-1).mean()


class DistillationLoss(Module):
    """Train a small model to imitate a big one's soft predictions.

    A teacher's full output distribution says far more than the label does:
    "this 7 looks somewhat like a 1" is knowledge the hard label destroys.
    Hinton called it dark knowledge, and it is why a student trained on teacher
    outputs can beat the same student trained on ground truth.

    Both sets of logits are divided by a TEMPERATURE first. Raising T flattens
    the distributions, magnifying the small probabilities where the useful
    similarity structure lives -- at T=1 they are so close to zero that they
    contribute almost no gradient. The ``T**2`` factor restores the gradient
    magnitude that the division removed, so T can be changed without retuning
    the learning rate.

    Hinton, Vinyals & Dean (2015).
    """

    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, target=None):
        T = self.temperature
        soft_teacher = F.softmax(Tensor._wrap(teacher_logits).detach() * (1.0 / T),
                                 axis=-1)
        soft_student = F.log_softmax(student_logits * (1.0 / T), axis=-1)
        soft = KLDivLoss()(soft_student, soft_teacher) * (T * T)
        if target is None:
            return soft
        hard = CrossEntropyLoss()(student_logits, target)
        return soft * self.alpha + hard * (1.0 - self.alpha)


class HingeEmbeddingLoss(Module):
    """``max(0, 1 - y*z)`` over +/-1 targets: the SVM loss, as an nn module.

    Zero once a sample is correct AND a margin clear, so confident-correct
    points contribute no gradient at all and the boundary is decided only by the
    points near it. Contrast with log-loss, which never stops nudging.
    """

    def __init__(self, margin=1.0, squared=False):
        super().__init__()
        self.margin = margin
        self.squared = squared

    def forward(self, scores, target):
        t = Tensor._wrap(target).detach()
        margins = (1.0 * self.margin - t * scores).relu()
        return (margins * margins).mean() if self.squared else margins.mean()


class MarginRankingLoss(Module):
    """``max(0, -y*(s1 - s2) + margin)``: learn an ORDER, not a value.

    The right loss when only relative order matters -- search ranking,
    preference data. It never asks a score to hit a target, only to beat the
    other one by a margin, which is a far weaker and more achievable demand.
    """

    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin

    def forward(self, score1, score2, target):
        t = Tensor._wrap(target).detach()
        return (-t * (score1 - score2) + self.margin).relu().mean()


# ---------------------------------------------------------------------------
# segmentation: losses that optimise overlap rather than per-pixel accuracy
# ---------------------------------------------------------------------------

class DiceLoss(Module):
    """1 - Dice coefficient: optimise OVERLAP, not per-pixel correctness.

    Segmenting a small tumour in a large scan, pixel-wise cross-entropy is
    maximised by predicting "background" everywhere -- 99.9% accurate, entirely
    useless. The Dice coefficient measures overlap instead::

        Dice = 2*|X and Y| / (|X| + |Y|)

    and normalising by the region SIZE makes it indifferent to how small the
    target is. That is what makes it the standard loss for medical segmentation.

    The soft version uses probabilities directly rather than thresholding, which
    keeps it differentiable. ``smooth`` avoids 0/0 when both are empty -- and
    also makes an empty prediction on an empty target score perfectly, which is
    the desired behaviour.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs, target):
        t = Tensor._wrap(target).detach()
        intersection = (probs * t).sum()
        total = probs.sum() + t.sum()
        return 1.0 - (intersection * 2.0 + self.smooth) / (total + self.smooth)


class TverskyLoss(Module):
    """Dice, with false positives and false negatives priced separately.

    Dice weights a missed pixel and a spurious one equally. Often they are not:
    missing a tumour is worse than flagging healthy tissue. Tversky adds the
    dial::

        T = TP / (TP + alpha*FP + beta*FN)

    ``alpha = beta = 0.5`` recovers Dice exactly. Note the doubling below: the
    ratio is scaled to ``2*TP / (2*TP + FP + FN)`` before smoothing is applied,
    because otherwise ``smooth`` would enter at half the weight it has in
    ``DiceLoss`` and the two would silently disagree at alpha = beta = 0.5.

    Raising ``beta`` punishes misses harder, trading precision for recall -- the
    same decision discussed in ``nupyml.metrics``, made differentiable.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, probs, target):
        t = Tensor._wrap(target).detach()
        tp = (probs * t).sum()
        fp = (probs * (1.0 - t)).sum()
        fn = ((1.0 - probs) * t).sum()
        num = tp * 2.0
        den = tp * 2.0 + fp * (2.0 * self.alpha) + fn * (2.0 * self.beta)
        return 1.0 - (num + self.smooth) / (den + self.smooth)


class IoULoss(Module):
    """1 - Jaccard index: ``|X and Y| / |X or Y|``.

    The same motivation as Dice and a stricter measure -- IoU is always <= Dice,
    and punishes disagreement more sharply. It is also usually the metric
    segmentation is REPORTED with, so optimising it directly avoids the mismatch
    of training on one thing and being judged on another.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs, target):
        t = Tensor._wrap(target).detach()
        intersection = (probs * t).sum()
        union = probs.sum() + t.sum() - intersection
        return 1.0 - (intersection + self.smooth) / (union + self.smooth)


# ---------------------------------------------------------------------------
# regression
# ---------------------------------------------------------------------------

class LogCoshLoss(Module):
    """``log(cosh(x))``: MSE near zero, MAE far away, smooth everywhere.

    Huber does the same job with a hard switch at ``delta``; log-cosh has no
    parameter and no kink, being twice differentiable everywhere -- which
    matters if anything downstream wants a Hessian.

    Computed as ``|x| + log1p(exp(-2|x|)) - log(2)`` rather than literally:
    ``cosh`` overflows for x beyond ~710, and the model is most wrong exactly
    when x is large.
    """

    def forward(self, pred, target):
        t = Tensor._wrap(target).detach()
        d = pred - t
        a = d.abs()
        return (a + ((a * -2.0).exp() + 1.0).log() - np.log(2.0)).mean()


class SmoothL1Loss(Module):
    """Huber with ``delta = beta``, scaled so the quadratic region has slope 1.

    The detection-community spelling of the same idea. The scaling matters in
    practice: bounding-box regression targets are small, so the quadratic region
    is where all the action is, and its gradient scale sets the effective
    learning rate.
    """

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        t = Tensor._wrap(target).detach()
        d = (pred - t).abs()
        quad = d * d * (0.5 / self.beta)
        lin = d - 0.5 * self.beta
        return Tensor.where(d.data < self.beta, quad, lin).mean()


class QuantileLoss(Module):
    """Pinball loss: fit a QUANTILE instead of the mean.

    ``max(q*(y - pred), (q-1)*(y - pred))`` -- asymmetric on purpose. Under-
    predicting costs ``q`` per unit and over-predicting costs ``1-q``, so the
    minimiser is the q-th quantile rather than the mean.

    Train one head per quantile and you have a prediction INTERVAL from a model
    that never mentions a distribution. ``q = 0.5`` gives the median, i.e. MAE.
    """

    def __init__(self, quantile=0.5):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred, target):
        t = Tensor._wrap(target).detach()
        d = t - pred
        q = self.quantile
        return Tensor.where((d.data >= 0), d * q, d * (q - 1.0)).mean()


class TweedieLoss(Module):
    """Deviance for the Tweedie family, over a log-link prediction.

    Insurance claims and similar targets are a spike at exactly zero plus a
    continuous positive tail -- neither gaussian nor Poisson. The Tweedie family
    with ``1 < power < 2`` (compound Poisson-gamma) models exactly that, and
    this is its deviance as a differentiable loss.

    See ``nupyml.linear_model.TweedieRegressor`` for how ``power`` selects the
    distribution.
    """

    def __init__(self, power=1.5):
        super().__init__()
        self.power = power

    def forward(self, log_pred, target):
        t = Tensor._wrap(target).detach()
        p = self.power
        mu = log_pred.exp()
        a = t * (mu ** (1.0 - p)) * (1.0 / (1.0 - p))
        b = (mu ** (2.0 - p)) * (1.0 / (2.0 - p))
        return (-a + b).mean() * 2.0


# ---------------------------------------------------------------------------
# adversarial
# ---------------------------------------------------------------------------

class WassersteinLoss(Module):
    """The critic loss that made GANs trainable.

    A standard GAN discriminator outputs a probability and is trained with
    log-loss. When it wins -- which it does easily -- its output saturates, its
    gradient vanishes, and the generator receives no signal. Training collapses,
    and the failure is silent.

    WGAN replaces the discriminator with a CRITIC that outputs an unbounded
    score, and maximises ``E[critic(real)] - E[critic(fake)]``. This estimates
    the Wasserstein (earth-mover) distance, which -- unlike JS divergence -- is
    finite and has a usable gradient even when the two distributions do not
    overlap at all. That is the whole point: the generator always knows which
    way to move.

    The estimate is only valid if the critic is 1-Lipschitz, which is what
    ``gradient_penalty`` below enforces.
    """

    def forward(self, critic_real, critic_fake, for_critic=True):
        if for_critic:
            # the critic MAXIMISES the gap, so minimise its negation
            return critic_fake.mean() - critic_real.mean()
        return -critic_fake.mean()


def gradient_penalty(critic, real, fake, rng=None):
    """Penalise the critic's gradient norm away from 1 (WGAN-GP).

    The Wasserstein estimate needs a 1-Lipschitz critic, meaning its gradient
    norm never exceeds 1. The original WGAN clipped weights to force it, which
    also crippled the critic's capacity.

    WGAN-GP instead notes that an optimal critic has gradient norm EXACTLY 1
    almost everywhere along the path between the distributions, and penalises
    deviation from it -- sampled on random interpolations between real and fake
    points, which is where the constraint actually binds.

    Gulrajani et al. (2017).
    """
    from ..utils import check_random_state
    rng = check_random_state(rng)
    real_d = real.data if isinstance(real, Tensor) else np.asarray(real)
    fake_d = fake.data if isinstance(fake, Tensor) else np.asarray(fake)
    eps = rng.uniform(size=(len(real_d),) + (1,) * (real_d.ndim - 1))
    mixed = Tensor(eps * real_d + (1 - eps) * fake_d, requires_grad=True)
    score = critic(mixed).sum()
    score.backward()
    grad = mixed.grad.reshape(len(real_d), -1)
    norm = np.sqrt((grad ** 2).sum(axis=1) + 1e-12)
    return float(np.mean((norm - 1.0) ** 2))


__all__ = ["MSELoss", "MAELoss", "HuberLoss", "CrossEntropyLoss", "NLLLoss",
           "BCELoss", "BCEWithLogitsLoss",
           "FocalLoss", "LabelSmoothingCrossEntropy", "KLDivLoss",
           "DistillationLoss", "HingeEmbeddingLoss", "MarginRankingLoss",
           "DiceLoss", "TverskyLoss", "IoULoss",
           "LogCoshLoss", "SmoothL1Loss", "QuantileLoss", "TweedieLoss",
           "WassersteinLoss", "gradient_penalty"]
