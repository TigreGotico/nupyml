"""Additional modern losses, extending the already-rich ``losses`` module.

These are autograd ``Module``s like the rest: ``forward`` returns a scalar
``Tensor`` you can ``.backward()`` through. They target the long-tail /
imbalance, robust-regression, count, and supervised-contrastive settings the
base zoo did not yet cover.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


def _as_int(target):
    return np.asarray(target.data if isinstance(target, Tensor) else target,
                      dtype=np.int64)


def _as_float(target):
    return np.asarray(target.data if isinstance(target, Tensor) else target,
                      dtype=np.float64)


class ClassBalancedLoss(Module):
    """Reweight classes by their EFFECTIVE number of samples (Cui et al., 2019).

    THE PROBLEM WITH 1/frequency
    ----------------------------
    Long-tailed data tempts you to weight each class by ``1/count``. But samples
    within a class OVERLAP -- the 1000th cat photo adds far less new information
    than the 2nd -- so raw frequency over-weights rare classes. The "effective
    number" ``(1 - beta^n) / (1 - beta)`` models this saturation: it grows with
    ``n`` but flattens out, so a class with 10 samples and one with 10000 are not
    treated as 1000x apart. The per-class weight is ``(1 - beta) / (1 - beta^n)``.

    ``beta`` near 1 (e.g. 0.999) makes the effect strong; ``beta=0`` recovers no
    reweighting. Wraps cross-entropy or focal loss.
    """

    def __init__(self, samples_per_class, beta=0.999, gamma=0.0):
        super().__init__()
        n = np.asarray(samples_per_class, dtype=np.float64)
        eff = 1.0 - np.power(beta, n)
        w = (1.0 - beta) / np.maximum(eff, 1e-12)
        self.weights = w / w.sum() * len(w)          # normalise to mean 1
        self.gamma = gamma

    def forward(self, logits, target):
        target = _as_int(target)
        logp = F.log_softmax(logits, axis=-1)
        n = logp.shape[0]
        logp_t = logp[np.arange(n), target]
        modulator = (1.0 - logp_t.exp()) ** self.gamma if self.gamma else 1.0
        w = Tensor(self.weights[target])
        return (-(w * modulator * logp_t)).mean()


class BalancedSoftmaxLoss(Module):
    """Correct the softmax for the TRAINING class prior (Ren et al., 2020).

    Under class imbalance the softmax bakes in the training frequencies, so at
    test time (where classes are balanced) it over-predicts the head classes.
    Balanced softmax simply ADDS ``log(prior_c)`` to each class logit before the
    ordinary cross-entropy::

        loss = CE( logits + log(prior),  target )

    This is the Bayes-consistent adjustment: it cancels the training prior so the
    learned scores reflect the likelihood, not the frequency. One line, no tuning,
    and it often beats elaborate reweighting on long-tailed benchmarks.
    """

    def __init__(self, samples_per_class):
        super().__init__()
        prior = np.asarray(samples_per_class, float)
        self.log_prior = np.log(prior / prior.sum() + 1e-12)

    def forward(self, logits, target):
        target = _as_int(target)
        adjusted = logits + Tensor(self.log_prior)   # broadcast add to every row
        logp = F.log_softmax(adjusted, axis=-1)
        n = logp.shape[0]
        return (-logp[np.arange(n), target]).mean()


class PolyLoss(Module):
    """Cross-entropy plus a first-order polynomial correction (Leng et al., 2022).

    Poly-1 adds a single term to cross-entropy::

        loss = CE + epsilon * (1 - p_true)

    Expanding CE as a Taylor series in ``(1 - p_true)`` shows its leading term is
    exactly ``(1 - p_true)``; PolyLoss lets you tune that leading coefficient
    directly with one hyperparameter, which reliably beats plain CE by a small
    margin across tasks. Positive ``epsilon`` up-weights not-yet-confident
    examples (like a gentle focal loss); negative ``epsilon`` down-weights them.
    """

    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, target):
        target = _as_int(target)
        logp = F.log_softmax(logits, axis=-1)
        n = logp.shape[0]
        logp_t = logp[np.arange(n), target]
        p_t = logp_t.exp()
        ce = -logp_t
        return (ce + self.epsilon * (1.0 - p_t)).mean()


class FocalTverskyLoss(Module):
    """Tversky index, focused on hard regions (Abraham & Khan, 2019).

    THE SEGMENTATION IMBALANCE
    --------------------------
    In segmentation the object is a few pixels among a sea of background, so
    Dice/IoU (which weight false positives and false negatives equally) still let
    the easy background dominate. The Tversky index weights them SEPARATELY::

        TI = TP / (TP + alpha*FP + beta*FN)

    -- raise ``beta`` to punish missed object pixels (recall) over false alarms.
    Focal-Tversky then raises the shortfall to a power, ``(1 - TI) ** gamma``, to
    concentrate training on the images the model is getting wrong. Operates on
    per-sample sigmoid probabilities of the positive class.
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, target):
        y = Tensor(_as_float(target))
        p = logits.sigmoid()
        tp = (p * y).sum()
        fp = (p * (1.0 - y)).sum()
        fn = ((1.0 - p) * y).sum()
        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return (1.0 - ti) ** self.gamma


class GHMLoss(Module):
    """Gradient Harmonizing Mechanism for classification (Li et al., 2019).

    THE INSIGHT
    -----------
    Focal loss down-weights easy examples by a fixed formula; GHM asks the data
    instead. It bins examples by their GRADIENT NORM ``g = |p - y|`` and
    down-weights each example by how CROWDED its bin is -- because a bin holding
    thousands of examples (the trivial easy ones, and separately the noisy
    outliers with huge gradients) is over-represented in the total gradient.
    Dividing each example's loss by its bin's gradient DENSITY equalises the
    contribution across the gradient spectrum, taming BOTH the easy majority and
    the harmful outliers at once -- something focal loss does not do.

    The bin weights are computed from the current predictions and treated as
    constants (they modulate the loss, they are not differentiated through).
    Binary classification on logits.
    """

    def __init__(self, bins=10):
        super().__init__()
        self.bins = bins

    def forward(self, logits, target):
        y = _as_float(target)
        p = logits.sigmoid()
        g = np.abs(p.data - y)                        # gradient norm per example
        n = len(g)
        edges = np.linspace(0, 1.0 + 1e-6, self.bins + 1)
        weights = np.ones(n)
        for b in range(self.bins):
            mask = (g >= edges[b]) & (g < edges[b + 1])
            count = mask.sum()
            if count > 0:
                weights[mask] = n / (count * self.bins)   # inverse bin density
        w = Tensor(weights)
        # weighted binary cross-entropy
        eps = 1e-7
        bce = -(Tensor(y) * (p + eps).log()
                + Tensor(1.0 - y) * ((1.0 - p) + eps).log())
        return (w * bce).mean()


class WingLoss(Module):
    """Wing loss for regression of small targets (Feng et al., 2018).

    Built for facial-landmark regression, where most errors are TINY and a few are
    large. L2 ignores the small errors (their gradient vanishes near zero) yet is
    dominated by the large ones; L1 treats all equally. Wing loss is LOGARITHMIC
    for small errors -- so it keeps pushing on the many near-misses that matter --
    and LINEAR for large ones, so outliers do not explode it::

        wing(x) = w * ln(1 + |x|/eps)      if |x| < w
                = |x| - C                   otherwise

    with ``C`` chosen to make the two pieces meet continuously. ``w`` sets the
    switch point, ``eps`` the curvature of the log region.
    """

    def __init__(self, w=10.0, eps=2.0):
        super().__init__()
        self.w = w
        self.eps = eps
        self.C = w - w * np.log(1 + w / eps)

    def forward(self, pred, target):
        diff = (pred - Tensor(_as_float(target))).abs()
        small = self.w * (1.0 + diff / self.eps).log()
        large = diff - self.C
        mask = (diff.data < self.w).astype(np.float64)   # constant selector
        return (small * Tensor(mask) + large * Tensor(1.0 - mask)).mean()


class PoissonNLLLoss(Module):
    """Negative log-likelihood for COUNT targets under a Poisson model.

    When the target is a count (events per interval, clicks, defects), squared
    error is wrong -- the variance grows with the mean, and predictions can go
    negative. The Poisson NLL respects that::

        log_input=True :  loss = exp(input) - target * input
        log_input=False:  loss = input - target * log(input)

    With ``log_input=True`` (the safe default) the network outputs ``log(rate)``,
    so the rate ``exp(input)`` is automatically positive and the loss is stable.
    This is the count-data analogue of using cross-entropy for classes.
    """

    def __init__(self, log_input=True, eps=1e-8):
        super().__init__()
        self.log_input = log_input
        self.eps = eps

    def forward(self, pred, target):
        y = Tensor(_as_float(target))
        if self.log_input:
            return (pred.exp() - y * pred).mean()
        return (pred - y * (pred + self.eps).log()).mean()


class SupConLoss(Module):
    """Supervised contrastive loss (Khosla et al., 2020).

    THE GENERALISATION OF InfoNCE
    -----------------------------
    Self-supervised contrastive learning has exactly ONE positive per anchor (an
    augmented view of itself). Supervised contrastive uses the LABELS: every other
    sample of the same class is a positive. For each anchor it pulls ALL its
    same-class embeddings together and pushes the rest away::

        L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(z_i·z_p / T)
                                               / sum_{a != i} exp(z_i·z_a / T) )

    Averaging over multiple positives gives a far more robust embedding geometry
    than a single-positive loss, and it consistently beats cross-entropy as a
    pre-training objective for the encoder. ``temperature`` sharpens the
    similarities. Embeddings are L2-normalised so the dot product is a cosine.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        labels = _as_int(labels)
        # L2-normalise so dot products are cosines (differentiable)
        norm = (embeddings * embeddings).sum(axis=1).sqrt().reshape(-1, 1)
        z = embeddings / (norm + 1e-12)
        sim = (z @ z.T) * (1.0 / self.temperature)
        n = sim.shape[0]
        # mask out self-similarity with a large negative constant on the diagonal
        neg_inf = np.where(np.eye(n) > 0, -1e9, 0.0)
        logp = F.log_softmax(sim + Tensor(neg_inf), axis=-1)   # over all a != i
        same = (labels[:, None] == labels[None, :]).astype(np.float64)
        np.fill_diagonal(same, 0.0)                  # positives exclude self
        pos_counts = same.sum(axis=1)
        valid = pos_counts > 0
        # mean log-prob over each anchor's positives, averaged over valid anchors
        pos_logp = (logp * Tensor(same)).sum(axis=1) / Tensor(np.maximum(pos_counts, 1))
        loss = -(pos_logp * Tensor(valid.astype(np.float64)))
        return loss.sum() / max(1, int(valid.sum()))


__all__ = ["ClassBalancedLoss", "BalancedSoftmaxLoss", "PolyLoss",
           "FocalTverskyLoss", "GHMLoss", "WingLoss", "PoissonNLLLoss",
           "SupConLoss"]
