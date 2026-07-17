"""Metric learning: losses that shape the embedding space itself.

THE SHIFT IN GOAL
-----------------
A classifier learns a decision boundary between a FIXED set of classes. Metric
learning learns a MAP into a space where distance means similarity -- and then
never needs to be told the classes at all.

That distinction is what these losses are for. A face-recognition system cannot
be a classifier: it must handle people it never trained on. So instead of
learning "who is this?", learn "are these two the same person?" -- a question
whose answer is a distance, and which transfers to strangers. The same reasoning
covers retrieval, deduplication, recommendation and one-shot learning.

THE PROGRESSION
---------------
Each loss here fixes the previous one's weakness:

``ContrastiveLoss``
    Pairs. Pull matches together, push non-matches apart beyond a margin. Simple
    -- and it demands an ABSOLUTE distance, which is a stronger claim than
    necessary.
``TripletLoss``
    Triples. Only requires that a positive be CLOSER than a negative, by a
    margin. Relative, not absolute -- a far weaker demand, and therefore easier
    to satisfy. Its weakness is that most triples become trivially satisfied and
    stop teaching, which is why mining matters (see below).
``NPairsLoss``
    One anchor against MANY negatives at once, rather than one. More signal per
    step, and no mining needed to find a hard negative -- take lots and let
    softmax find the hard one.
``InfoNCELoss``
    N-pairs with a temperature, and the insight that "positive" can be a
    distorted copy of the anchor. That removes labels entirely, which is what
    makes modern self-supervised learning work.
``ArcFaceLoss`` / ``CosFaceLoss``
    Go back to a classifier, but impose the margin on the ANGLE, so the learned
    features are directly usable as an embedding. Best of both.

WHY NORMALIZE
-------------
Most of these compare directions, not magnitudes: embeddings are projected onto
the unit sphere and similarity is a dot product (equivalently, a cosine). This
stops the model from cheating by inflating the norm of easy examples, and makes
every distance comparable.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from ..utils import check_random_state
from .module import Module, Parameter
from . import init


def _l2_normalize(x, eps=1e-12):
    """Project rows onto the unit sphere so only direction matters."""
    norm = (x * x).sum(axis=-1, keepdims=True) ** 0.5
    return x / (norm + eps)


def _pairwise_sq_dists(a, b):
    """Squared euclidean distances between every row of a and every row of b.

    Uses ``||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2`` so the whole matrix comes
    from one matmul rather than a Python loop -- the same expansion that lets
    kernel methods work in terms of inner products.

    Clipped at zero: the expansion can produce a tiny negative from round-off
    when two points coincide, and the sqrt of that is nan.
    """
    aa = (a * a).sum(axis=-1, keepdims=True)
    bb = (b * b).sum(axis=-1, keepdims=True).transpose()
    return (aa - a @ b.transpose() * 2.0 + bb).relu()


class ContrastiveLoss(Module):
    """Pull matching pairs together; push non-matching pairs past a margin.

    ::

        L = y * d^2  +  (1 - y) * max(0, margin - d)^2

    with ``y = 1`` for a matching pair. Read the two halves separately:

    * Matches pay ``d^2`` -- an unbounded pull, so they are drawn together
      indefinitely. There is no "close enough".
    * Non-matches pay only until they are ``margin`` apart, then nothing. Without
      that cutoff the loss would push every non-match to infinity, which is both
      impossible and pointless: it is enough to be distinguishable.

    The margin is why this does not collapse to mapping everything to one point,
    which would otherwise be a perfect score on the first term.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, target):
        y = Tensor._wrap(target).detach()
        diff = emb1 - emb2
        d_sq = (diff * diff).sum(axis=-1)
        d = (d_sq + 1e-12) ** 0.5
        pull = y * d_sq
        push = (1.0 - y) * ((self.margin - d).relu() ** 2)
        return (pull + push).mean()


class TripletLoss(Module):
    """Require the positive to be closer than the negative, by a margin.

    ::

        L = max(0, d(a, p) - d(a, n) + margin)

    Only the ORDER is constrained, never the absolute distance -- a much weaker
    demand than ``ContrastiveLoss`` makes, and correspondingly easier to satisfy
    while still producing a useful space.

    THE MINING PROBLEM
    ------------------
    Once training is underway, a random triple almost always already satisfies
    the constraint, so its loss is exactly 0 and its gradient is exactly 0. Most
    of every batch teaches nothing, and training grinds to a halt while appearing
    to converge.

    Fixing this means MINING -- deliberately selecting triples that violate the
    constraint. ``BatchHardTripletLoss`` below does it inside the batch, which is
    the standard practical answer.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = ((anchor - positive) ** 2).sum(axis=-1)
        d_neg = ((anchor - negative) ** 2).sum(axis=-1)
        return (d_pos - d_neg + self.margin).relu().mean()


class BatchHardTripletLoss(Module):
    """Triplet loss that mines its own hardest triples inside each batch.

    For every sample, take the FURTHEST positive and the NEAREST negative in the
    batch -- the two that most violate the constraint. So instead of hoping a
    random triple is informative, every triple is the hardest one available.

    The batch is doing double duty: it is both the training data and the mining
    pool, which is why this needs no offline mining pass and no separate index.

    Mining the very hardest can destabilise training if the data has label noise
    (the "hardest positive" may simply be mislabelled), which is what
    semi-hard mining exists to soften.

    Hermans, Beyer & Leibe (2017), "In Defense of the Triplet Loss".
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        labels = np.asarray(labels if not isinstance(labels, Tensor)
                            else labels.data)
        d_sq = _pairwise_sq_dists(embeddings, embeddings)
        d = (d_sq + 1e-12) ** 0.5
        same = labels[:, None] == labels[None, :]
        eye = np.eye(len(labels), dtype=bool)
        pos_mask = same & ~eye              # same label, not itself
        neg_mask = ~same
        # hardest positive: the furthest one. Mask others to -inf before max.
        hardest_pos = Tensor.where(pos_mask, d, Tensor(np.full(d.shape, -np.inf))
                                   ).max(axis=1)
        # hardest negative: the nearest one. Mask others to +inf before min.
        hardest_neg = Tensor.where(neg_mask, d, Tensor(np.full(d.shape, np.inf))
                                   ).min(axis=1)
        return (hardest_pos - hardest_neg + self.margin).relu().mean()


class NPairsLoss(Module):
    """One anchor against many negatives at once, as a softmax.

    A triplet compares the positive to ONE negative per step. N-pairs compares it
    to every other class in the batch simultaneously, turning the problem into
    cross-entropy over similarities::

        L = -log( exp(a.p) / sum_k exp(a.n_k) )

    More gradient signal per step, and no mining required: softmax already
    concentrates the gradient on whichever negative is hardest, automatically.
    """

    def forward(self, anchors, positives):
        sim = anchors @ positives.transpose()
        target = np.arange(sim.shape[0])          # anchor i matches positive i
        logp = F.log_softmax(sim, axis=-1)
        return -logp[target, target].mean()


class InfoNCELoss(Module):
    """Contrastive loss with a temperature. The engine of self-supervised learning.

    Given an anchor and its positive, classify which of the batch's items is the
    positive::

        L = -log( exp(sim(a, p) / tau) / sum_k exp(sim(a, k) / tau) )

    THE IDEA THAT MADE IT MATTER
    ----------------------------
    The "positive" need not come from a label. Take an image, apply two random
    distortions, and declare the two views a positive pair -- everything else in
    the batch is a negative. No labels anywhere, yet the model must learn what
    survives a distortion, which is exactly the semantic content. That is
    SimCLR, and it is why representation learning stopped needing labelled data.

    THE TEMPERATURE
    ---------------
    ``tau`` sharpens the softmax and matters more than it looks. Large tau
    treats all negatives roughly alike; small tau concentrates the entire
    gradient on the hardest negative, which learns fine distinctions but is
    unstable and punishes false negatives (batch items that ARE semantically the
    same but unlabelled) severely. 0.07-0.5 is the usual range.

    Also known as NT-Xent. Oord et al. (2018); Chen et al. (2020).
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchors, positives):
        a = _l2_normalize(anchors)
        p = _l2_normalize(positives)
        sim = (a @ p.transpose()) * (1.0 / self.temperature)
        target = np.arange(sim.shape[0])
        logp = F.log_softmax(sim, axis=-1)
        return -logp[target, target].mean()


class CosFaceLoss(Module):
    """Softmax classification with an additive margin on the cosine.

    Plain softmax only asks that the correct logit be largest, which leaves the
    classes merely separable -- adjacent, with no gap. For an embedding to
    generalise to unseen identities it must be DISCRIMINATIVE: a gap between
    classes, not just a boundary.

    So normalise both weights and features (making every logit a cosine), then
    subtract a margin from the true class's score::

        logit_true = s * (cos(theta) - m)

    The true class is now handicapped, and the only way to win is by a margin --
    which carves a gap around each class.

    The scale ``s`` is needed because cosines live in [-1, 1], and a softmax over
    such a narrow range can never become confident; ``s`` re-expands them.

    Wang et al. (2018).
    """

    def __init__(self, in_features, n_classes, scale=30.0, margin=0.35, rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self.weight = Parameter(init.xavier_uniform((in_features, n_classes), rng))
        self.scale = scale
        self.margin = margin

    def forward(self, features, target):
        target = np.asarray(target if not isinstance(target, Tensor)
                            else target.data, dtype=np.int64)
        cos = _l2_normalize(features) @ _l2_normalize(self.weight.transpose()).transpose()
        onehot = np.eye(cos.shape[1])[target]
        logits = (cos - Tensor(onehot) * self.margin) * self.scale
        return CrossEntropyOnLogits(logits, target)


class ArcFaceLoss(Module):
    """CosFace's cousin: the margin goes on the ANGLE, not the cosine.

    ::

        logit_true = s * cos(theta + m)

    Why this is better than subtracting m from the cosine: cosine is non-linear
    in the angle, so a fixed cosine margin corresponds to a different angular gap
    depending on where you are on the arc -- tight near theta=0, generous near
    theta=90. An additive ANGULAR margin is constant everywhere, giving a uniform
    geodesic gap on the hypersphere. That is the entire ArcFace argument.

    ``arccos`` is never called. Expand instead::

        cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)

    with ``sin(theta) = sqrt(1 - cos^2(theta))``. Cheaper, and it keeps the whole
    thing differentiable through operations the tape already knows.

    Deng et al. (2019).
    """

    def __init__(self, in_features, n_classes, scale=30.0, margin=0.5, rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self.weight = Parameter(init.xavier_uniform((in_features, n_classes), rng))
        self.scale = scale
        self.margin = margin

    def forward(self, features, target):
        target = np.asarray(target if not isinstance(target, Tensor)
                            else target.data, dtype=np.int64)
        cos = _l2_normalize(features) @ _l2_normalize(self.weight.transpose()).transpose()
        cos = cos.clip(-1 + 1e-7, 1 - 1e-7)          # keep the sqrt real
        sin = (1.0 - cos * cos + 1e-12) ** 0.5
        cos_with_margin = cos * np.cos(self.margin) - sin * np.sin(self.margin)
        onehot = Tensor(np.eye(cos.shape[1])[target])
        logits = (onehot * cos_with_margin + (1.0 - onehot) * cos) * self.scale
        return CrossEntropyOnLogits(logits, target)


class CenterLoss(Module):
    """Pull each sample toward a learned centre for its class.

    Softmax separates classes but says nothing about how TIGHT each one is, so
    the features spread out into wedges. Center loss adds the missing term::

        L = 0.5 * sum ||x_i - c_{y_i}||^2

    with the centres learned alongside the weights. Used ADDED to cross-entropy,
    never alone -- alone, its minimum is every centre and every feature at the
    same point. Softmax does the separating, center loss does the compacting.

    Wen et al. (2016).
    """

    def __init__(self, n_classes, n_features, rng=None):
        super().__init__()
        rng = check_random_state(rng)
        self.centers = Parameter(rng.normal(scale=0.1,
                                            size=(n_classes, n_features)))

    def forward(self, features, target):
        target = np.asarray(target if not isinstance(target, Tensor)
                            else target.data, dtype=np.int64)
        chosen = self.centers[target]
        diff = features - chosen
        return (diff * diff).sum(axis=-1).mean() * 0.5


def CrossEntropyOnLogits(logits, target):
    """Shared tail for the margin losses: log-softmax then pick the true class."""
    logp = F.log_softmax(logits, axis=-1)
    n = logp.shape[0]
    return -logp[np.arange(n), target].mean()


__all__ = ["ContrastiveLoss", "TripletLoss", "BatchHardTripletLoss",
           "NPairsLoss", "InfoNCELoss", "CosFaceLoss", "ArcFaceLoss",
           "CenterLoss"]
