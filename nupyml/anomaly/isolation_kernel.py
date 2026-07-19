"""A similarity that ADAPTS to the data's density (Ting et al., 2018)."""
import numpy as np
from scipy.spatial.distance import cdist
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


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


__all__ = ["IsolationKernel"]
