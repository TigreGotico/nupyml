"""Make a classifier say how much it does NOT know (Sensoy et al., 2018)."""
import numpy as np
from ..autograd import Tensor
from .module import Module


class EvidentialLoss(Module):
    """Make a classifier say how much it does NOT know (Sensoy et al., 2018).

    Softmax always produces a confident-looking distribution, even for nonsense
    inputs. Evidential deep learning treats the network's outputs as EVIDENCE for a
    Dirichlet distribution over the class probabilities: more total evidence means a
    sharper Dirichlet (confident), little evidence means one near uniform
    (uncertain). Training minimises the Bayes risk of the sum-of-squares (Brier)
    loss under that Dirichlet plus a penalty on evidence for WRONG classes -- so the
    model learns to withhold evidence when unsure. ``uncertainty`` returns ``K/S``,
    high when the model has seen nothing like the input (useful for OOD).
    """

    def __init__(self, n_classes, lam=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.lam = lam

    def forward(self, evidence, target):
        # evidence >= 0 (e.g. relu/softplus of logits); alpha = evidence + 1
        evidence = Tensor._wrap(evidence).relu()
        alpha = evidence + 1.0
        S = alpha.sum(axis=1, keepdims=True)
        p = alpha / S
        onehot = np.eye(self.n_classes)[np.asarray(target)]
        y = Tensor(onehot)
        err = ((y - p) * (y - p)).sum(axis=1)              # Brier error term
        var = (p * (1.0 - p) / (S + 1.0)).sum(axis=1)      # Dirichlet variance term
        # penalise evidence assigned to the wrong classes (drives them to uniform)
        wrong = (evidence * (1.0 - y)).sum(axis=1)
        return (err + var).mean() + self.lam * wrong.mean()

    @staticmethod
    def uncertainty(evidence):
        evidence = np.maximum(np.asarray(evidence), 0.0)
        alpha = evidence + 1.0
        K = evidence.shape[1]
        return K / alpha.sum(axis=1)                       # vacuity: high = unknown


__all__ = ["EvidentialLoss"]
