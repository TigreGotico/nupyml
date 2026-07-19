"""Slope One: predict from the average PAIRWISE rating DIFFERENCE (Lemire, 2005)."""
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


class SlopeOne(BaseEstimator):
    """Slope One: predict from the average PAIRWISE rating DIFFERENCE (Lemire, 2005).

    THE APPEALINGLY SIMPLE IDEA
    ---------------------------
    For every pair of items, compute how much higher users rate item i than item j
    on average -- the "deviation" ``dev(i,j)``. To predict user u's rating of item
    i, take each item j the user HAS rated, add ``dev(i,j)`` to their rating
    ``r(u,j)``, and average::

        pred(u,i) = mean over rated j of ( r(u,j) + dev(i,j) )

    No latent factors, no training loop -- just precomputed pairwise deviations --
    yet it is competitive with far heavier methods and trivially updatable online.
    The "slope one" name: it fits predictors of the form ``f(x)=x+b`` (slope fixed
    at one, only the offset learned).
    """

    def fit(self, triples):
        R, M = _matrix(triples)
        self.R_, self.M_ = R, M
        ni = R.shape[1]
        self.dev_ = np.zeros((ni, ni))
        self.freq_ = np.zeros((ni, ni))
        for u in range(R.shape[0]):
            rated = np.where(M[u])[0]
            for a in rated:
                for b in rated:
                    self.dev_[a, b] += R[u, a] - R[u, b]
                    self.freq_[a, b] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            self.dev_ = np.where(self.freq_ > 0, self.dev_ / self.freq_, 0.0)
        self.global_mean_ = R[M].mean()
        return self

    def predict(self, user, item):
        if user >= self.R_.shape[0] or item >= self.R_.shape[1]:
            return float(self.global_mean_)
        rated = np.where(self.M_[user])[0]
        num = den = 0.0
        for j in rated:
            f = self.freq_[item, j]
            if f > 0:
                num += (self.R_[user, j] + self.dev_[item, j]) * f
                den += f
        return float(num / den) if den > 0 else float(self.global_mean_)


__all__ = ["SlopeOne"]
