"""Model compression: pruning, quantization, low-rank factorization, distillation.

A trained network is usually far larger than it needs to be. These shrink it for
deployment -- fewer weights (pruning), fewer bits per weight (quantization),
smaller matrices (low-rank), or a smaller network entirely (distillation) -- while
keeping most of the accuracy.
"""
import numpy as np

from ..autograd import Tensor
from ..autograd import functional as F


def magnitude_prune(model, sparsity=0.5):
    """Zero the smallest-magnitude weights GLOBALLY (Han et al., 2015).

    THE LOTTERY-TICKET OBSERVATION
    ------------------------------
    Most weights in a trained network are tiny and contribute almost nothing --
    the important computation lives in a sparse subset. Magnitude pruning ranks
    ALL weights by ``|w|`` and zeros the bottom ``sparsity`` fraction, typically
    removing 80-90% with negligible accuracy loss (then fine-tuning recovers the
    rest). A global threshold (not per-layer) lets pruning fall where the network
    least needs capacity. Returns the fraction actually zeroed.
    """
    weights = [p for p in model.parameters() if p.data.ndim >= 2]   # skip biases
    allw = np.concatenate([np.abs(w.data).ravel() for w in weights])
    thresh = np.quantile(allw, sparsity)
    total = zeroed = 0
    for w in weights:
        mask = np.abs(w.data) >= thresh
        w.data *= mask                               # prune in place
        total += w.data.size
        zeroed += np.sum(~mask)
    return zeroed / total


def quantize_weights(model, bits=8):
    """Round each weight tensor to ``2**bits`` uniform levels (post-training).

    Storing weights as 8-bit integers instead of 32-bit floats is a 4x memory (and
    often speed) win. Post-training quantization maps each tensor's ``[min, max]``
    range onto ``2**bits`` evenly-spaced levels and snaps every weight to the
    nearest -- no retraining. The quantisation ERROR it injects is usually
    tolerable at 8 bits; below that, quantization-aware training is needed.
    Modifies weights in place and returns the mean absolute quantization error.
    """
    levels = 2 ** bits - 1
    err = 0.0
    n = 0
    for p in model.parameters():
        lo, hi = p.data.min(), p.data.max()
        scale = (hi - lo) / levels if hi > lo else 1.0
        q = np.round((p.data - lo) / scale) * scale + lo
        err += np.abs(q - p.data).sum()
        n += p.data.size
        p.data[...] = q
    return err / n


def low_rank_approximation(W, rank):
    """Factor a weight matrix ``W (m x n)`` into ``A (m x r) @ B (r x n)`` via SVD.

    A dense ``m x n`` layer costs ``m*n`` parameters and multiplies. If its weight
    matrix is approximately low-rank -- and trained layers often are -- keeping the
    top ``r`` singular components replaces it with two thin matrices costing
    ``r*(m+n)``, a large saving when ``r << min(m, n)``. The truncated SVD is the
    optimal rank-``r`` approximation in Frobenius norm (Eckart-Young). Returns
    ``(A, B)`` with ``A @ B`` approximating ``W``.
    """
    U, s, Vt = np.linalg.svd(np.asarray(W, float), full_matrices=False)
    A = U[:, :rank] * s[:rank]
    B = Vt[:rank]
    return A, B


class KnowledgeDistillation:
    """Train a small STUDENT to mimic a large TEACHER's soft predictions
    (Hinton et al., 2015).

    THE "DARK KNOWLEDGE"
    --------------------
    A teacher's full probability vector says more than the hard label: that a 7
    looks 30% like a 1 encodes similarity structure the one-hot target throws away.
    Distillation trains the student on the teacher's SOFTENED probabilities
    (divide logits by a temperature ``T`` to expose the small ones), optionally
    mixed with the true labels. The student learns the teacher's generalisation --
    reaching accuracy its own capacity could not from hard labels alone.

    ``fit`` trains the student network on inputs, using the teacher's logits as the
    soft target (via the existing DistillationLoss).
    """

    def __init__(self, student, teacher, temperature=4.0, alpha=0.5, lr=0.01,
                 epochs=30, batch_size=32, random_state=None):
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y=None):
        from .optim import Adam
        from .losses import DistillationLoss
        from ..utils import check_random_state
        rng = check_random_state(self.random_state)
        X = np.asarray(X, float)
        teacher_logits = self._teacher_logits(X)
        opt = Adam(self.student.parameters(), lr=self.lr)
        loss_fn = DistillationLoss(temperature=self.temperature, alpha=self.alpha)
        n = len(X)
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                b = perm[s:s + self.batch_size]
                opt.zero_grad()
                sl = self.student(Tensor(X[b]))
                tgt = None if y is None else np.asarray(y)[b]
                loss_fn(sl, Tensor(teacher_logits[b]), tgt).backward()
                opt.step()
        return self

    def _teacher_logits(self, X):
        out = self.teacher(Tensor(X))
        return out.data if isinstance(out, Tensor) else np.asarray(out)

    def predict(self, X):
        return self.student(Tensor(np.asarray(X, float))).data.argmax(axis=1)


__all__ = ["magnitude_prune", "quantize_weights", "low_rank_approximation",
           "KnowledgeDistillation"]
