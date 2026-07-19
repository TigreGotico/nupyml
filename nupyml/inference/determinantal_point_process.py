"""Determinantal point process: sample diverse subsets by repulsion."""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class DeterminantalPointProcess(BaseEstimator):
    """Sample DIVERSE subsets by repulsion (Kulesza & Taskar, 2012).

    Random sampling clumps; if you want a subset whose items are DIFFERENT from each
    other -- diverse search results, a representative summary, a spread-out minibatch
    -- you need repulsion. A determinantal point process makes the probability of a
    subset proportional to the DETERMINANT of the kernel submatrix on it, and a
    determinant is large exactly when the selected vectors are nearly orthogonal
    (dissimilar) and small when they are similar. So similar items rarely appear
    together. Exact sampling via the kernel's eigendecomposition. ``L`` is a PSD
    similarity kernel.
    """

    def __init__(self, L, random_state=None):
        self.L = np.asarray(L, float)
        self.random_state = random_state

    def sample(self):
        rng = check_random_state(self.random_state)
        vals, vecs = np.linalg.eigh(self.L)
        vals = np.clip(vals, 0, None)
        # phase 1: pick an eigenvector set with prob lambda/(lambda+1)
        included = rng.rand(len(vals)) < vals / (vals + 1.0)
        V = vecs[:, included]
        selected = []
        # phase 2: iteratively project out a chosen item
        while V.shape[1] > 0:
            probs = (V ** 2).sum(axis=1)
            probs = probs / probs.sum()
            i = rng.choice(len(probs), p=probs)
            selected.append(i)
            # find a vector in V with nonzero component at i, orthogonalise the rest
            j = np.argmax(np.abs(V[i]))
            Vj = V[:, j]
            V = np.delete(V, j, axis=1)
            if V.shape[1] > 0:
                V = V - np.outer(Vj, V[i] / Vj[i])
                # re-orthonormalise
                q, _ = np.linalg.qr(V) if V.shape[1] else (V, None)
                V = q
        return np.array(sorted(selected))


__all__ = ["DeterminantalPointProcess"]
