"""Pack normal data into the SMALLEST ball in a learned space (Ruff, 2018)."""
import numpy as np
from ..utils import check_array, check_random_state
from .detectors import _Detector


class DeepSVDD(_Detector):
    """Pack normal data into the SMALLEST ball in a learned space (Ruff, 2018).

    One-class SVM finds a boundary in a FIXED kernel space. Deep SVDD instead
    LEARNS the mapping (here a small neural net) so that normal points collapse
    toward a single centre ``c`` -- it minimises the mean squared distance of the
    embeddings to ``c``, shrinking the enclosing hypersphere. An anomaly, having no
    reason to map near ``c``, lands far away, so its distance from the centre is the
    anomaly score. The centre is fixed to the initial embedding mean (a nonzero
    ``c`` is essential -- ``c=0`` with bias-free layers admits the trivial all-zero
    collapse).
    """

    def __init__(self, hidden=8, epochs=60, lr=0.05, contamination=0.1,
                 random_state=None):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X):
        from ..nn import Linear
        from ..autograd import Tensor
        X = check_array(X)
        rng = check_random_state(self.random_state)
        self.l1 = Linear(X.shape[1], self.hidden, rng=rng)
        self.l2 = Linear(self.hidden, self.hidden, rng=rng)
        xt = Tensor(X)
        self.center_ = self._embed(xt).data.mean(axis=0)   # fix c = initial mean
        c = Tensor(self.center_)
        params = list(self.l1.parameters()) + list(self.l2.parameters())
        for _ in range(self.epochs):
            emb = self._embed(xt)
            diff = emb - c
            loss = (diff * diff).sum(axis=1).mean()
            for p in params:
                p.zero_grad()
            loss.backward()
            for p in params:
                p.data -= self.lr * p.grad
        self._set_threshold(self.decision_function(X))
        return self

    def _embed(self, xt):
        return self.l2(self.l1(xt).relu())

    def decision_function(self, X):
        from ..autograd import Tensor
        X = check_array(X)
        emb = self._embed(Tensor(X)).data
        return np.sqrt(((emb - self.center_) ** 2).sum(axis=1))   # distance to c


__all__ = ["DeepSVDD"]
