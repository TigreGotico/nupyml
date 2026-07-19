"""Message passing between points to pick exemplars."""
import warnings
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from ..base import BaseEstimator, ClusterMixin, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


class AffinityPropagation(BaseEstimator, ClusterMixin):
    """Message passing between points to pick exemplars."""

    def __init__(self, damping=0.5, max_iter=200, convergence_iter=15,
                 preference=None):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference

    def fit(self, X, y=None):
        X = check_array(X)
        n = len(X)
        S = -cdist(X, X, "sqeuclidean")
        pref = self.preference if self.preference is not None else np.median(S)
        np.fill_diagonal(S, pref)
        A = np.zeros((n, n))
        R = np.zeros((n, n))
        d = self.damping
        last_exemplars = None
        stable = 0
        converged = False
        for it in range(self.max_iter):
            # responsibilities
            AS = A + S
            idx_max = AS.argmax(axis=1)
            first_max = AS[np.arange(n), idx_max]
            AS[np.arange(n), idx_max] = -np.inf
            second_max = AS.max(axis=1)
            R_new = S - first_max[:, None]
            R_new[np.arange(n), idx_max] = S[np.arange(n), idx_max] - second_max
            R = d * R + (1 - d) * R_new
            # availabilities
            Rp = np.maximum(R, 0)
            np.fill_diagonal(Rp, np.diag(R))
            A_new = Rp.sum(axis=0)[None, :] - Rp
            dA = np.diag(A_new).copy()
            A_new = np.minimum(A_new, 0)
            np.fill_diagonal(A_new, dA)
            A = d * A + (1 - d) * A_new
            exemplars = np.where(np.diag(A) + np.diag(R) > 0)[0]
            if last_exemplars is not None and np.array_equal(exemplars,
                                                             last_exemplars):
                stable += 1
                if stable >= self.convergence_iter:
                    converged = True
                    break
            else:
                stable = 0
            last_exemplars = exemplars
        if len(exemplars) == 0:   # degenerate: everything in one cluster
            exemplars = np.array([int(np.argmax(np.diag(S)))])
        self.converged_ = converged
        if not converged:
            warnings.warn(
                "Affinity propagation did not converge; the exemplars and "
                "labels may be degenerate. Try raising damping or max_iter, "
                "or lowering preference.", UserWarning, stacklevel=2)
        self.cluster_centers_indices_ = exemplars
        self.cluster_centers_ = X[exemplars]
        self.labels_ = cdist(X, self.cluster_centers_).argmin(axis=1)
        self.n_iter_ = it + 1
        return self

    def predict(self, X):
        check_is_fitted(self, "cluster_centers_")
        return cdist(check_array(X), self.cluster_centers_).argmin(axis=1)


__all__ = ["AffinityPropagation"]
