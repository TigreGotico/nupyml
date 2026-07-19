"""Collaborative filtering via simultaneous user AND item clustering"""
import numpy as np
from ..base import BaseEstimator


def _matrix(triples):
    triples = [(int(u), int(i), float(r)) for u, i, r in triples]
    nu = max(u for u, _, _ in triples) + 1
    ni = max(i for _, i, _ in triples) + 1
    R = np.zeros((nu, ni)); M = np.zeros((nu, ni), bool)
    for u, i, r in triples:
        R[u, i] = r; M[u, i] = True
    return R, M


class CoClusteringCF(BaseEstimator):
    """Collaborative filtering via simultaneous user AND item clustering
    (George & Merugu, 2005).

    THE IDEA
    --------
    Assign users to ``n_user_clusters`` and items to ``n_item_clusters``; a
    (user-cluster, item-cluster) pair is a CO-CLUSTER with its own average rating.
    A prediction combines three levels of averages so it degrades gracefully when
    data is sparse::

        pred(u,i) = cocluster_avg(gu, gi)
                  + (user_avg(u)      - user_cluster_avg(gu))
                  + (item_avg(i)      - item_cluster_avg(gi))

    -- the co-cluster gives the base level, and the two correction terms add back
    the user's and item's own biases relative to their clusters. Fast, and a
    strong baseline that needs no gradient training. Clusters are found by k-means
    on the users' and items' rating vectors.
    """

    def __init__(self, n_user_clusters=3, n_item_clusters=3, random_state=None):
        self.n_user_clusters = n_user_clusters
        self.n_item_clusters = n_item_clusters
        self.random_state = random_state

    def fit(self, triples):
        from ..cluster import KMeans
        R, M = _matrix(triples)
        self.R_, self.M_ = R, M
        self.global_mean_ = R[M].mean()
        nu, ni = R.shape
        # cluster users by their rating rows, items by their rating columns
        self.ug_ = KMeans(n_clusters=min(self.n_user_clusters, nu),
                          random_state=self.random_state).fit(R).labels_
        self.ig_ = KMeans(n_clusters=min(self.n_item_clusters, ni),
                          random_state=self.random_state).fit(R.T).labels_

        def avg(mask2d):
            return R[mask2d].mean() if mask2d.any() else self.global_mean_
        self.user_avg_ = np.array([R[u, M[u]].mean() if M[u].any()
                                   else self.global_mean_ for u in range(nu)])
        self.item_avg_ = np.array([R[M[:, i], i].mean() if M[:, i].any()
                                   else self.global_mean_ for i in range(ni)])
        self.uc_avg_ = {g: avg(M & (self.ug_ == g)[:, None]) for g in np.unique(self.ug_)}
        self.ic_avg_ = {g: avg(M & (self.ig_ == g)[None, :]) for g in np.unique(self.ig_)}
        self.cocluster_ = {}
        for gu in np.unique(self.ug_):
            for gi in np.unique(self.ig_):
                mask = M & (self.ug_ == gu)[:, None] & (self.ig_ == gi)[None, :]
                self.cocluster_[(gu, gi)] = avg(mask)
        return self

    def predict(self, user, item):
        if user >= self.R_.shape[0] or item >= self.R_.shape[1]:
            return float(self.global_mean_)
        gu, gi = self.ug_[user], self.ig_[item]
        return float(self.cocluster_[(gu, gi)]
                     + (self.user_avg_[user] - self.uc_avg_[gu])
                     + (self.item_avg_[item] - self.ic_avg_[gi]))


__all__ = ["CoClusteringCF"]
