"""Neighborhood collaborative filtering and SLIM.

The factorization models compress the rating matrix into dense latent vectors.
These take the opposite, MEMORY-BASED route: keep the ratings and predict from
similar rows or columns directly. Neighborhood methods are transparent ("we
recommended this because people like you liked it") and need no gradient
training; SLIM keeps that item-item structure but LEARNS the similarities by
regression instead of fixing them to cosine.

All three take ``triples`` -- an iterable of ``(user, item, rating)`` -- to match
the factorization estimators.
"""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


def _build_matrix(triples):
    triples = [(int(u), int(i), float(r)) for u, i, r in triples]
    n_users = max(u for u, _, _ in triples) + 1
    n_items = max(i for _, i, _ in triples) + 1
    R = np.zeros((n_users, n_items))
    M = np.zeros((n_users, n_items), dtype=bool)
    for u, i, r in triples:
        R[u, i] = r
        M[u, i] = True
    return R, M


def _cosine_sim(A):
    """Row-wise cosine similarity of a matrix (zeros where a row is empty)."""
    norm = np.linalg.norm(A, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    An = A / norm
    return An @ An.T


class UserBasedCF(BaseEstimator):
    """Predict a rating from what SIMILAR USERS thought of the item.

    THE ORIGINAL RECOMMENDER IDEA
    -----------------------------
    "People who agreed with you in the past will agree with you now." Represent
    each user by their row of ratings, find the ``k`` users most similar (cosine,
    on mean-centred ratings so a generous rater and a harsh one can still match in
    SHAPE), and predict an unseen rating as their similarity-weighted average
    deviation from their own means::

        r(u,i) = mean_u + sum_v sim(u,v) * (r(v,i) - mean_v) / sum_v |sim(u,v)|

    Transparent and training-free -- but it must scan users at predict time, and
    it struggles when the matrix is very sparse (few co-rated items make the
    similarities noisy). That sparsity/scale limitation is exactly what pushed the
    field toward factorization.
    """

    def __init__(self, k=20):
        self.k = k

    def fit(self, triples):
        self.R_, self.M_ = _build_matrix(triples)
        counts = self.M_.sum(axis=1)
        counts[counts == 0] = 1
        self.user_means_ = (self.R_ * self.M_).sum(axis=1) / counts
        centred = (self.R_ - self.user_means_[:, None]) * self.M_
        self.sim_ = _cosine_sim(centred)
        np.fill_diagonal(self.sim_, 0.0)       # a user is not their own neighbour
        return self

    def predict(self, user, item):
        raters = np.where(self.M_[:, item])[0]
        if len(raters) == 0:
            return float(self.user_means_[user])
        sims = self.sim_[user, raters]
        order = np.argsort(-np.abs(sims))[:self.k]
        raters, sims = raters[order], sims[order]
        denom = np.abs(sims).sum()
        if denom == 0:
            return float(self.user_means_[user])
        dev = self.R_[raters, item] - self.user_means_[raters]
        return float(self.user_means_[user] + (sims * dev).sum() / denom)


class ItemBasedCF(BaseEstimator):
    """Predict from how the user rated SIMILAR ITEMS. The Amazon classic.

    WHY ITEM-ITEM WON IN PRACTICE
    -----------------------------
    User-user similarities change every time anyone rates anything and there are
    usually far more users than items, so they are expensive and unstable. ITEM
    similarities are far more stable (an item's rating pattern barely moves as one
    more user rates it) and can be PRECOMPUTED offline. At predict time you only
    look at the handful of items THIS user already rated -- fast and cache-friendly.
    That is why item-based CF, not user-based, powered the early large-scale
    recommenders.

    Prediction is the similarity-weighted average of the user's own ratings on the
    items most similar to the target.

    Sarwar et al. (2001).
    """

    def __init__(self, k=20):
        self.k = k

    def fit(self, triples):
        self.R_, self.M_ = _build_matrix(triples)
        item_counts = self.M_.sum(axis=0)
        item_counts[item_counts == 0] = 1
        self.item_means_ = (self.R_ * self.M_).sum(axis=0) / item_counts
        centred = ((self.R_ - self.item_means_[None, :]) * self.M_).T   # items x users
        self.sim_ = _cosine_sim(centred)
        np.fill_diagonal(self.sim_, 0.0)
        return self

    def predict(self, user, item):
        rated = np.where(self.M_[user])[0]
        if len(rated) == 0:
            return float(self.item_means_[item])
        sims = self.sim_[item, rated]
        order = np.argsort(-np.abs(sims))[:self.k]
        rated, sims = rated[order], sims[order]
        denom = np.abs(sims).sum()
        if denom == 0:
            return float(self.item_means_[item])
        return float((sims * self.R_[user, rated]).sum() / denom)


class SLIM(BaseEstimator):
    """Sparse LInear Method: LEARN the item-item weights instead of fixing them.

    THE STEP BEYOND ITEM-KNN
    ------------------------
    Item-based CF fixes item similarities to cosine and hopes they predict well.
    SLIM instead LEARNS an item-item weight matrix W so that each item's column is
    reconstructed from the OTHER items' columns::

        minimise  ||R - R W||^2  +  beta*||W||^2  +  lambda*||W||_1
        subject to  diag(W) = 0,  W >= 0

    The score for (user, item j) is then ``R[user] @ W[:, j]`` -- a learned
    weighted vote from the items the user interacted with. The L1 penalty makes W
    SPARSE (each item is explained by a few others, so it stays fast and
    interpretable), the non-negativity keeps weights as "endorsements", and the
    zero diagonal stops an item trivially predicting itself. It routinely beats
    both KNN and plain factorization on top-N ranking.

    Ning & Karypis (2011). Columns fit independently by coordinate descent.
    """

    def __init__(self, l1=0.001, l2=0.1, max_iter=100, tol=1e-4):
        self.l1 = l1
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, triples):
        self.R_, self.M_ = _build_matrix(triples)
        R = self.R_
        n_items = R.shape[1]
        gram = R.T @ R                          # precompute item-item Gram matrix
        W = np.zeros((n_items, n_items))
        diag = np.diag(gram)
        for j in range(n_items):                # reconstruct column j from the rest
            w = W[:, j]
            for _ in range(self.max_iter):
                w_old = w.copy()
                for kk in range(n_items):
                    if kk == j:
                        continue                # keep the diagonal zero
                    # coordinate descent residual for weight w[kk]
                    rho = gram[kk, j] - (gram[kk] @ w) + gram[kk, kk] * w[kk]
                    # soft-threshold (L1) with L2 ridge and non-negativity
                    val = rho - self.l1
                    w[kk] = max(val, 0.0) / (diag[kk] + self.l2 + 1e-12)
                if np.abs(w - w_old).max() < self.tol:
                    break
            W[:, j] = w
        self.W_ = W
        return self

    def predict(self, user, item):
        return float(self.R_[user] @ self.W_[:, item])

    def recommend(self, user, n=10, exclude=None):
        scores = self.R_[user] @ self.W_
        if exclude is not None:
            scores[list(exclude)] = -np.inf
        return np.argsort(scores)[::-1][:n]


__all__ = ["UserBasedCF", "ItemBasedCF", "SLIM"]
