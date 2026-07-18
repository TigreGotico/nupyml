"""Loss zoo v2: noise-robust classification, robust/relative regression, a
listwise ranking loss, and a self-supervised regulariser.

All are autograd ``Module``s -- ``forward`` returns a scalar Tensor to backprop.
They join the large existing loss set with families it did not cover.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


def _int(t):
    return np.asarray(t.data if isinstance(t, Tensor) else t, dtype=np.int64)


def _float(t):
    return np.asarray(t.data if isinstance(t, Tensor) else t, dtype=np.float64)


class SymmetricCrossEntropy(Module):
    """CE + reverse-CE, robust to NOISY labels (Wang et al., 2019).

    Cross-entropy over-trusts labels: a mislabelled example produces a huge
    gradient that the network dutifully fits, memorising the noise. Symmetric CE
    adds a REVERSE term -- the cross-entropy of the label given the PREDICTION --
    which saturates for confident-but-wrong cases and stops them dominating::

        SCE = alpha * CE(p, y) + beta * CE(y, p)

    The reverse term is bounded, so noisy labels can no longer drag training. A
    one-line robustness fix over plain CE.
    """

    def __init__(self, alpha=1.0, beta=1.0, clip=-4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.clip = clip

    def forward(self, logits, target):
        target = _int(target)
        logp = F.log_softmax(logits, axis=-1)
        n, k = logp.shape
        ce = -logp[np.arange(n), target].mean()
        p = logp.exp()
        # reverse CE: treat the (one-hot) label's log as clipped constant
        onehot = np.eye(k)[target]
        log_y = np.clip(np.log(onehot + 1e-12), self.clip, 0.0)
        rce = -(p * Tensor(log_y)).sum(axis=1).mean()
        return self.alpha * ce + self.beta * rce


class GeneralizedCrossEntropy(Module):
    """A noise-robust interpolation between CE and MAE (Zhang & Sabuncu, 2018).

    ``L_q = (1 - p_true^q) / q`` interpolates between cross-entropy (``q -> 0``,
    accurate but noise-sensitive) and mean-absolute-error-on-probabilities
    (``q = 1``, robust but slow). A middle ``q`` (e.g. 0.7) keeps CE's fast early
    learning while bounding the gradient a wrong label can produce -- the tunable
    robustness knob CE lacks.
    """

    def __init__(self, q=0.7):
        super().__init__()
        self.q = q

    def forward(self, logits, target):
        target = _int(target)
        p = F.softmax(logits, axis=-1)
        n = p.shape[0]
        p_true = p[np.arange(n), target]
        return ((1.0 - p_true ** self.q) / self.q).mean()


class AsymmetricLoss(Module):
    """Asymmetric loss for long-tailed MULTI-LABEL classification (Ben-Baruch, 2020).

    In multi-label problems each image has a few positive labels and hundreds of
    negatives, so the easy negatives dominate. Asymmetric loss focuses on positives
    and negatives DIFFERENTLY: a larger focusing ``gamma_neg`` for negatives (down-
    weight the easy ones hard) than ``gamma_pos`` for positives, plus a probability
    SHIFT that fully discards very-easy negatives. This asymmetry is what makes it
    the strong default for multi-label with extreme negative imbalance. Operates on
    per-label sigmoid probabilities.
    """

    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, clip=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits, target):
        y = Tensor(_float(target))
        p = logits.sigmoid()
        pm = p if self.clip <= 0 else (p - self.clip)   # shift down easy negatives
        pm = pm.relu() * (1 - y) + p * y                # only shift the negatives
        loss_pos = y * ((1 - p) ** self.gamma_pos) * (p + 1e-8).log()
        loss_neg = (1 - y) * (pm ** self.gamma_neg) * ((1 - p) + 1e-8).log()
        return -(loss_pos + loss_neg).mean()


class CharbonnierLoss(Module):
    """A smooth, robust L1 (Charbonnier / pseudo-Huber): ``sqrt((y-p)^2 + eps^2)``.

    L2 is smooth but outlier-sensitive; L1 is robust but has a kink at zero (bad
    gradient there). Charbonnier is L1's smooth cousin -- quadratic near zero (clean
    gradient) and linear far away (outlier-robust) -- controlled by one ``eps``. The
    standard reconstruction loss in image super-resolution and optical flow.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - Tensor(_float(target))
        return ((diff * diff + self.eps ** 2).sqrt()).mean()


class MSLELoss(Module):
    """Mean squared error in LOG space -- penalise relative, not absolute, error.

    For targets spanning orders of magnitude (counts, prices, populations) squared
    error is dominated by the large values. MSLE takes the error of ``log(1+y)``,
    so being off by 10% costs the same whether the true value is 10 or 10000 -- it
    measures RELATIVE error, and it penalises UNDER-prediction more than over.
    Requires non-negative targets.
    """

    def forward(self, pred, target):
        y = Tensor(_float(target))
        lp = (pred.relu() + 1.0).log()
        ly = (y + 1.0).log()
        return ((lp - ly) ** 2).mean()


class SMAPELoss(Module):
    """Symmetric mean absolute percentage error -- a bounded relative loss.

    Plain MAPE blows up when the target is near zero and is asymmetric (penalises
    over- and under-prediction unequally). SMAPE divides the absolute error by the
    AVERAGE of the magnitudes, bounding it in [0, 2] and treating over/under
    symmetrically -- the standard forecasting-competition metric, here as a
    differentiable loss.
    """

    def forward(self, pred, target):
        y = Tensor(_float(target))
        num = (pred - y).abs()
        denom = pred.abs() + y.abs() + 1e-8
        return (num / denom).mean()


class ListMLELoss(Module):
    """Listwise ranking loss: the Plackett-Luce likelihood of the correct order
    (Xia et al., 2008).

    Pairwise ranking losses (RankNet) look at pairs; listwise losses model the
    WHOLE permutation. ListMLE maximises the probability, under the Plackett-Luce
    model, of drawing items in their true relevance order: at each position, the
    next item should have the highest score among those not yet placed::

        L = -sum_i [ s_{(i)} - logsumexp(s_{(i)}, s_{(i+1)}, ...) ]

    where ``(i)`` is the item at rank ``i`` in the ideal order. Optimising the
    entire ordering at once often beats pairwise losses on ranking metrics.
    ``scores`` and ``relevance`` are per-query 1-D arrays.
    """

    def forward(self, scores, relevance):
        rel = _float(relevance)
        order = np.argsort(-rel)                       # ideal descending order
        s = scores[order] if isinstance(scores, Tensor) else Tensor(scores)[order]
        # log P(order) = sum_i [ s_i - logsumexp(s_i..s_n) ]  (reverse cumulative)
        n = s.shape[0]
        loss = Tensor(0.0)
        # logsumexp of the tail via a numerically stable manual pass
        for i in range(n):
            tail = s[i:]
            m = tail.data.max()
            lse = (tail - Tensor(m)).exp().sum().log() + m
            loss = loss + (lse - s[i])
        return loss / n


class VICRegLoss(Module):
    """Variance-Invariance-Covariance regularisation (Bardes et al., 2022).

    A self-supervised loss for two augmented views that avoids collapse WITHOUT
    negatives or a target network -- purely by three explicit terms on the two
    views' embeddings ``za, zb``:

    * **Invariance** -- MSE between the paired views (they should agree).
    * **Variance** -- a hinge keeping each embedding dimension's std above 1, so
      the representation cannot collapse to a constant.
    * **Covariance** -- push the off-diagonal covariance to zero, so dimensions
      carry non-redundant information.

    The variance term is the anti-collapse mechanism SimCLR needs negatives for.
    """

    def __init__(self, sim_coef=25.0, var_coef=25.0, cov_coef=1.0):
        super().__init__()
        self.sim_coef = sim_coef
        self.var_coef = var_coef
        self.cov_coef = cov_coef

    def _var_cov(self, z):
        n, d = z.shape
        zc = z - z.mean(axis=0)
        std = (zc.var(axis=0) + 1e-4).sqrt()
        var_loss = (1.0 - std).relu().mean()           # hinge: keep std >= 1
        cov = (zc.T @ zc) * (1.0 / (n - 1))
        eye = np.eye(d)
        cov_loss = ((cov * Tensor(1.0 - eye)) ** 2).sum() * (1.0 / d)
        return var_loss, cov_loss

    def forward(self, za, zb):
        za, zb = Tensor._wrap(za), Tensor._wrap(zb)
        inv = ((za - zb) ** 2).mean()
        va, ca = self._var_cov(za)
        vb, cb = self._var_cov(zb)
        return (self.sim_coef * inv + self.var_coef * (va + vb)
                + self.cov_coef * (ca + cb))


__all__ = ["SymmetricCrossEntropy", "GeneralizedCrossEntropy", "AsymmetricLoss",
           "CharbonnierLoss", "MSLELoss", "SMAPELoss", "ListMLELoss", "VICRegLoss"]
