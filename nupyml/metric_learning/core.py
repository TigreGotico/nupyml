"""Neighbourhood Components Analysis and Large Margin Nearest Neighbours."""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


class NCA(BaseEstimator, TransformerMixin):
    """Learn a linear transform that makes leave-one-out kNN work.

    THE DIFFERENTIABLE TRICK
    ------------------------
    kNN accuracy is what you want to maximise, but it is a step function of the
    transform -- non-differentiable, nothing to descend. NCA's move is to replace
    HARD neighbour assignment with a SOFT one: point ``i`` picks point ``j`` as
    its neighbour with a probability that decays with their transformed distance
    (a softmax over distances). Under that soft rule the expected number of
    correctly-classified points becomes a smooth function of the transform ``L``,
    which gradient ascent can optimise directly.

    So NCA optimises a differentiable stand-in for the exact objective -- the same
    manoeuvre as replacing 0/1 loss with cross-entropy. Maximise it and you get a
    projection under which same-class points cluster and kNN succeeds.

    A BONUS: DIMENSIONALITY REDUCTION
    ---------------------------------
    Make ``L`` rectangular (``n_components`` < input dim) and NCA becomes a
    SUPERVISED linear dimensionality reducer -- it finds the low-dimensional
    projection that best preserves class neighbourhoods, which is often far more
    useful for visualisation than the unsupervised PCA.

    Goldberger, Roweis, Hinton & Salakhutdinov (2005).
    """

    def __init__(self, n_components=None, max_iter=100, learning_rate=0.01,
                 random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        n, d = X.shape
        k = self.n_components or d
        L = rng.normal(0, 0.1, size=(k, d))
        same = (y[:, None] == y[None, :])           # same-class mask

        for _ in range(self.max_iter):
            LX = X @ L.T                            # transformed points
            # soft neighbour probabilities: p_ij = softmax over -||Lx_i - Lx_j||^2
            diff = LX[:, None, :] - LX[None, :, :]
            dist2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(dist2, np.inf)         # a point is not its own neighbour
            expd = np.exp(-dist2 - np.max(-dist2, axis=1, keepdims=True))
            P = expd / (expd.sum(axis=1, keepdims=True) + 1e-12)
            # p_i = probability i is correctly classified = sum of p_ij over
            # same-class j. The objective is sum_i p_i
            p_i = np.sum(P * same, axis=1)

            # gradient of the soft LOO objective w.r.t. L (the NCA gradient)
            grad = np.zeros_like(L)
            for i in range(n):
                # weighted outer products: pull same-class, push different-class
                weighted = P[i][:, None] * (X[i] - X)
                term = p_i[i] * (P[i][:, None] * (X[i] - X)).T @ (X[i] - X)
                same_term = ((P[i] * same[i])[:, None] * (X[i] - X)).T @ (X[i] - X)
                grad += L @ (term - same_term)
            L += self.learning_rate * 2 * grad / n

        self.components_ = L
        self.n_features_out_ = k
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        return check_array(X) @ self.components_.T


class LMNN(BaseEstimator, TransformerMixin):
    """Large Margin Nearest Neighbours: pull targets in, push impostors out.

    THE TWO FORCES
    --------------
    For each point, fix its ``k`` same-label "target neighbours" (its friends).
    Then shape the metric with two competing pulls:

    * PULL each point toward its target neighbours -- same-label points should be
      close.
    * PUSH away any different-label "impostor" that has intruded within the
      target neighbours' radius PLUS a margin -- different-label points should be
      farther than your friends, by a safety gap.

    Only impostors that actually violate the margin contribute, exactly like an
    SVM's support vectors: points already correctly placed exert no force. This is
    the large-margin principle applied to a distance instead of a hyperplane, and
    it is why LMNN pairs so naturally with kNN -- it optimises precisely the
    local neighbourhood structure kNN relies on.

    The objective is convex in the Mahalanobis matrix ``M`` (a semidefinite
    program in the full treatment); this version does projected gradient descent
    on ``L`` for clarity.

    Weinberger & Saul (2009).
    """

    def __init__(self, k=3, mu=0.5, max_iter=100, learning_rate=1e-3,
                 random_state=None):
        self.k = k
        self.mu = mu                    # pull/push balance
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        from scipy.spatial.distance import cdist
        X, y = check_X_y(X, y)
        rng = check_random_state(self.random_state)
        n, d = X.shape
        L = np.eye(d)

        # target neighbours are fixed up front, in the ORIGINAL space: the same-
        # label nearest neighbours each point should be pulled toward
        targets = np.zeros((n, self.k), dtype=int)
        D0 = cdist(X, X)
        for i in range(n):
            same = np.where(y == y[i])[0]
            same = same[same != i]
            if len(same):
                order = same[np.argsort(D0[i, same])]
                targets[i] = order[:self.k] if len(order) >= self.k else \
                    np.pad(order, (0, self.k - len(order)), mode="edge")

        for _ in range(self.max_iter):
            LX = X @ L.T
            grad = np.zeros_like(L)
            for i in range(n):
                for j in targets[i]:
                    dij = LX[i] - LX[j]
                    # the pull: shrink distance to target neighbours
                    grad += (1 - self.mu) * np.outer(dij, X[i] - X[j])
                    d_target = dij @ dij
                    # the push: any different-label point inside the margin
                    for l in range(n):
                        if y[l] == y[i]:
                            continue
                        dil = LX[i] - LX[l]
                        d_imp = dil @ dil
                        # margin violated => this impostor exerts a push
                        if d_target + 1 > d_imp:
                            grad += self.mu * (np.outer(dij, X[i] - X[j])
                                               - np.outer(dil, X[i] - X[l]))
            L -= self.learning_rate * L @ grad / n

        self.components_ = L
        self.n_features_out_ = d
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        return check_array(X) @ self.components_.T


__all__ = ["NCA", "LMNN"]
