"""Supervised contrastive loss (Khosla et al., 2020)."""
import numpy as np
from ..autograd import Tensor
from ..autograd import functional as F
from .module import Module


def _as_int(target):
    return np.asarray(target.data if isinstance(target, Tensor) else target,
                      dtype=np.int64)


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


__all__ = ["SupConLoss"]
