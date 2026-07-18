"""Losses v5: adaptive robustness, IoU surrogates, proper scores, listwise ranking,
and proxy metric learning.

Five more losses. Barron's loss has a shape parameter that INTERPOLATES the common
robust losses. Lovász-softmax is a differentiable surrogate for the IoU metric.
CRPS is a proper score for probabilistic forecasts. ListNet ranks a whole LIST at
once. Proxy-anchor learns a metric against learnable class proxies, combining the
speed of a classification loss with the geometry of a pairwise one.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


def _abs(t):
    return t.relu() + (t * (-1.0)).relu()


class BarronLoss(Module):
    """ONE robust loss with a tunable shape (Barron, 2019).

    L2, Charbonnier, Cauchy, Geman-McClure, Welsch -- the robust losses are usually
    presented as separate choices, and picking one is guesswork. Barron's general
    loss is a single family with a continuous shape parameter ``alpha`` that
    RECOVERS all of them: ``alpha=2`` is squared error, ``alpha=1`` is a smooth L1
    (Charbonnier), ``alpha=0`` is Cauchy, ``alpha -> -inf`` is Welsch. Lower alpha
    means an outlier's influence saturates sooner. Because alpha is just a number,
    it can even be LEARNED, letting the model adapt its own robustness. ``c`` is the
    scale at which residuals start to be treated as outliers.
    """

    def __init__(self, alpha=1.0, c=1.0):
        super().__init__()
        self.alpha = alpha
        self.c = c

    def forward(self, pred, target):
        r = (Tensor._wrap(pred) - Tensor._wrap(target)) * (1.0 / self.c)
        a = self.alpha
        if a == 2.0:
            rho = 0.5 * (r * r)
        elif a == 0.0:                                   # Cauchy / Lorentzian
            rho = (0.5 * (r * r) + 1.0).log()
        else:
            b = abs(a - 2.0)
            rho = (b / a) * (((r * r) / b + 1.0) ** (a / 2.0) - 1.0)
        return rho.mean()


def lovasz_hinge(logits, labels):
    """A differentiable surrogate for the IoU (Jaccard) loss (Berman et al., 2018).

    Segmentation is judged by intersection-over-union, but IoU is discrete and
    non-differentiable, so people train on per-pixel cross-entropy and hope it
    correlates. Lovász-softmax optimises IoU DIRECTLY: the Jaccard set function is
    submodular, and its LOVÁSZ EXTENSION is a convex, piecewise-linear function that
    IS differentiable -- computed by sorting the pixel errors and taking a dot
    product with the gradient of the sorted Jaccard loss. Minimising it improves IoU
    where cross-entropy plateaus, especially for small foreground objects. Binary
    hinge form here (``labels`` in {0,1}, ``logits`` real).
    """
    logits = Tensor._wrap(logits).reshape(-1)
    labels = np.asarray(labels, float).ravel()
    signs = 2.0 * labels - 1.0                            # {-1, +1}
    errors = 1.0 - logits * Tensor(signs)                 # hinge margins
    order = np.argsort(-errors.data)                      # descending
    gt_sorted = labels[order]
    grad = _lovasz_grad(gt_sorted)
    errors_sorted = errors[order]
    return (errors_sorted.relu() * Tensor(grad)).sum()


def _lovasz_grad(gt_sorted):
    # gradient of the Jaccard loss for the sorted ground truth (Berman et al.)
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - np.cumsum(gt_sorted)
    union = gts + np.cumsum(1 - gt_sorted)
    jaccard = 1.0 - intersection / np.maximum(union, 1e-12)
    if p > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def crps_ensemble(forecast_samples, y):
    """A PROPER score for probabilistic forecasts (Gneiting & Raftery, 2007).

    A point forecast cannot be rewarded for honest uncertainty. The Continuous
    Ranked Probability Score judges a whole predictive DISTRIBUTION against the
    single observed value, and it is PROPER: it is minimised in expectation only by
    the true distribution, so a model cannot game it by lying about its confidence.
    In the ensemble form used here it is
    ``mean|x_i - y| - 0.5 * mean|x_i - x_j|`` -- reward for being close to the
    outcome, penalty for being over-dispersed -- and it is fully differentiable, so
    it trains probabilistic (ensemble/sample) forecasters directly. ``forecast_samples``
    is ``(batch, n_samples)``; ``y`` is ``(batch,)``.
    """
    x = Tensor._wrap(forecast_samples)
    y = Tensor(np.asarray(y, float).reshape(-1, 1))
    n = x.shape[1]
    term1 = _abs(x - y).mean(axis=1)                      # E|X - y|
    xi = x.reshape(x.shape[0], n, 1)
    xj = x.reshape(x.shape[0], 1, n)
    term2 = _abs(xi - xj).mean(axis=2).mean(axis=1)       # E|X - X'|
    return (term1 - 0.5 * term2).mean()


class ListNetLoss(Module):
    """Rank a whole LIST at once, not pair by pair (Cao et al., 2007).

    Pointwise ranking losses ignore that ranking is about ORDER; pairwise losses
    (RankNet) consider two items at a time and can conflict. ListNet takes the whole
    list: it turns both the model's scores and the true relevance labels into a
    "top-one probability" distribution (a softmax over items -- the probability each
    item is ranked first) and minimises the cross-entropy between them. Optimising
    the list distribution directly tends to order the whole list better than
    stitching pairwise decisions together. ``scores`` and ``relevance`` are each one
    query's item vector.
    """

    def __init__(self):
        super().__init__()

    def forward(self, scores, relevance):
        scores = Tensor._wrap(scores).reshape(1, -1)
        rel = np.asarray(relevance, float).reshape(1, -1)
        target = _softmax_np(rel)                         # true top-1 distribution
        logp = F.log_softmax(scores, axis=1)
        return -(Tensor(target) * logp).sum()


def _softmax_np(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class ProxyAnchorLoss(Module):
    """Metric learning against learnable class PROXIES (Kim et al., 2020).

    Pairwise metric losses (triplet, contrastive) must mine informative pairs from
    a batch, which is slow and finicky. Proxy losses keep one learnable PROXY vector
    per class and compare samples to proxies instead of to each other -- as fast as
    a classification loss. Proxy-Anchor treats each proxy as an ANCHOR and, for every
    proxy, pulls its positive samples in and pushes negatives out with a smooth
    LogSumExp weighting that emphasises the hard ones -- recovering the data-to-data
    relations pure proxy-NCA loses, while keeping the fast convergence. Cosine
    similarities between L2-normalised embeddings and proxies.
    """

    def __init__(self, n_classes, dim, margin=0.1, alpha=32.0, rng=None):
        super().__init__()
        from ..utils import check_random_state
        from .module import Parameter
        r = check_random_state(rng)
        self.proxies = Parameter(r.randn(n_classes, dim) * 0.1)
        self.n_classes = n_classes
        self.margin = margin
        self.alpha = alpha

    def parameters(self):
        return [self.proxies]

    def forward(self, embeddings, labels):
        x = Tensor._wrap(embeddings)
        x = x / ((x * x).sum(axis=1, keepdims=True) ** 0.5 + 1e-8)
        p = self.proxies / ((self.proxies * self.proxies).sum(axis=1, keepdims=True)
                            ** 0.5 + 1e-8)
        sim = x @ p.transpose()                           # (batch, n_classes) cosine
        labels = np.asarray(labels)
        onehot = np.eye(self.n_classes)[labels]           # (batch, n_classes)
        pos = Tensor(onehot); neg = Tensor(1.0 - onehot)
        # positive term: pull samples toward their proxy (hard positives weighted)
        pos_exp = (-self.alpha * (sim - self.margin)).exp() * pos
        pos_term = (1.0 + pos_exp.sum(axis=0)).log()
        has_pos = onehot.sum(axis=0) > 0
        pos_loss = (pos_term * Tensor(has_pos.astype(float))).sum() \
            / max(has_pos.sum(), 1)
        # negative term: push non-matching samples away from each proxy
        neg_exp = (self.alpha * (sim + self.margin)).exp() * neg
        neg_loss = (1.0 + neg_exp.sum(axis=0)).log().sum() / self.n_classes
        return pos_loss + neg_loss


__all__ = ["BarronLoss", "lovasz_hinge", "crps_ensemble", "ListNetLoss",
           "ProxyAnchorLoss"]
