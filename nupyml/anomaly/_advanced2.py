"""Anomaly / OOD v2: oblique isolation, data-dependent kernels, deep one-class,
and post-hoc OOD scores.

Isolation forests isolate anomalies with few random cuts, but axis-parallel cuts
leave artefacts; the extended forest and the isolation kernel both cut
OBLIQUELY. Deep SVDD learns a representation in which normal data packs into a
tiny ball. The energy score turns any trained classifier into an OOD detector for
free -- no retraining, just a different reading of its logits.
"""
import numpy as np
from scipy.spatial.distance import cdist

from ..base import BaseEstimator
from ..utils import check_array, check_random_state
from .detectors import _Detector


class ExtendedIsolationForest(_Detector):
    """Isolation forest with OBLIQUE cuts instead of axis-parallel (Hariri, 2019).

    A standard isolation forest cuts one feature at a time, so its decision regions
    are boxes -- and on data that is not axis-aligned this leaves ghostly bands of
    artificially low/high anomaly score parallel to the axes. The extended version
    cuts with a RANDOM HYPERPLANE (a random normal direction and a random
    intercept) at each node, removing the bias. Everything else is identical: an
    anomaly is isolated by fewer cuts, so a SHORTER average path length means a
    higher anomaly score.
    """

    def __init__(self, n_estimators=100, max_samples=256, contamination=0.1,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

    def _build(self, X, depth, max_depth, rng):
        n = len(X)
        if depth >= max_depth or n <= 1:
            return {"size": n}
        normal = rng.randn(X.shape[1])                 # random hyperplane direction
        proj = X @ normal
        p = rng.uniform(proj.min(), proj.max())        # random intercept
        left = proj < p
        if left.all() or (~left).all():
            return {"size": n}
        return {"normal": normal, "p": p,
                "left": self._build(X[left], depth + 1, max_depth, rng),
                "right": self._build(X[~left], depth + 1, max_depth, rng)}

    @staticmethod
    def _c(n):
        if n <= 1:
            return 0.0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n

    def _path(self, x, node, depth):
        if "normal" not in node:
            return depth + self._c(node["size"])
        branch = node["left"] if x @ node["normal"] < node["p"] else node["right"]
        return self._path(x, branch, depth + 1)

    def fit(self, X):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        m = min(self.max_samples, len(X))
        max_depth = int(np.ceil(np.log2(max(m, 2))))
        self.trees_ = []
        for _ in range(self.n_estimators):
            sub = X[rng.choice(len(X), m, replace=False)]
            self.trees_.append(self._build(sub, 0, max_depth, rng))
        self._norm = self._c(m)
        self._set_threshold(self.decision_function(X))
        return self

    def decision_function(self, X):
        X = check_array(X)
        paths = np.array([[self._path(x, t, 0) for t in self.trees_] for x in X])
        avg = paths.mean(axis=1)
        return 2.0 ** (-avg / self._norm)              # short path -> high score


class IsolationKernel(BaseEstimator):
    """A similarity that ADAPTS to the data's density (Ting et al., 2018).

    Every fixed kernel (RBF, etc.) treats two points the same distance apart
    identically, wherever they sit. The isolation kernel makes similarity
    DATA-DEPENDENT: it repeatedly partitions space using random samples (a Voronoi
    partition around ``psi`` anchor points), and two points are similar to the
    degree that they keep landing in the SAME cell. In a dense region the cells are
    tiny, so points must be very close to count as similar; in a sparse region the
    cells are large, so the same gap counts as similar. This "sharper where it is
    crowded" behaviour is exactly what nearest-neighbour anomaly detection wants,
    and it needs no distance metric at all. ``transform`` gives the binary feature
    map whose dot product is the kernel.
    """

    def __init__(self, n_estimators=200, psi=8, random_state=None):
        self.n_estimators = n_estimators
        self.psi = psi
        self.random_state = random_state

    def fit(self, X):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        self.anchors_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(len(X), min(self.psi, len(X)), replace=False)
            self.anchors_.append(X[idx])               # partition = Voronoi(anchors)
        return self

    def transform(self, X):
        X = check_array(X)
        feats = np.zeros((len(X), self.n_estimators * self.psi))
        for i, anchors in enumerate(self.anchors_):
            cell = cdist(X, anchors).argmin(axis=1)    # nearest anchor = cell id
            feats[np.arange(len(X)), i * self.psi + cell] = 1.0
        return feats / np.sqrt(self.n_estimators)

    def similarity(self, X, Y=None):
        fx = self.transform(X)
        fy = fx if Y is None else self.transform(Y)
        return fx @ fy.T


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


def energy_score(logits, temperature=1.0):
    """Turn a classifier's logits into an OOD score (Liu et al., 2020).

    A softmax hides how confident a network really is -- it always sums to 1, so an
    out-of-distribution input can still get a spuriously high max-probability. The
    free energy ``-T * logsumexp(logits / T)`` does not: it is LOW for in-
    distribution inputs (some class fires strongly) and HIGH for OOD inputs (no
    class does), and it is theoretically aligned with the data density in a way the
    softmax score is not. Needs only the trained logits -- no retraining, no OOD
    data. Higher score = more out-of-distribution.
    """
    logits = np.asarray(logits, dtype=float)
    a = logits / temperature
    m = a.max(axis=1, keepdims=True)                   # stable logsumexp of a
    lse = m.ravel() + np.log(np.exp(a - m).sum(axis=1))
    return -temperature * lse                          # free energy = OOD score


__all__ = ["ExtendedIsolationForest", "IsolationKernel", "DeepSVDD",
           "energy_score"]
