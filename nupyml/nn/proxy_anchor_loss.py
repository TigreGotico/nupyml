"""Metric learning against learnable class PROXIES (Kim et al., 2020)."""
import numpy as np
from ..autograd import Tensor
from .module import Module


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


__all__ = ["ProxyAnchorLoss"]
