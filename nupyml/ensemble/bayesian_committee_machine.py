"""Bayesian committee machine: fuse regressors by their precision."""
import numpy as np

from ..base import BaseEstimator, RegressorMixin, clone
from ..utils import check_X_y, check_array


class BayesianCommitteeMachine(BaseEstimator, RegressorMixin):
    """Fuse regressors by their PRECISION, not a flat average (Tresp, 2000).

    Split a big regression across several experts (each trained on part of the data)
    and you must recombine them -- but a plain average trusts a wildly uncertain expert
    as much as a confident one. The Bayesian committee machine weights each expert's
    prediction by its INVERSE VARIANCE (precision) and corrects for the shared prior,
    so where an expert is unsure its vote fades and the confident experts dominate. It
    is the principled way to pool independent probabilistic regressors. Each base
    estimator must expose ``predict(X, return_std=True)``.
    """

    def __init__(self, estimators, prior_var=1.0):
        self.estimators = estimators
        self.prior_var = prior_var

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        rng = np.random.RandomState(0)
        parts = np.array_split(rng.permutation(len(X)), len(self.estimators))
        self.experts_ = []
        for est, idx in zip(self.estimators, parts):
            e = clone(est)
            e.fit(X[idx], y[idx])                          # each expert sees a shard
            self.experts_.append(e)
        return self

    def predict(self, X, return_std=False):
        X = check_array(X)
        M = len(self.experts_)
        prec = np.zeros(len(X)); mean_acc = np.zeros(len(X))
        for e in self.experts_:
            mu, sd = e.predict(X, return_std=True)
            p = 1.0 / (sd ** 2 + 1e-12)                    # expert precision
            prec += p; mean_acc += p * mu
        # subtract the (M-1) times over-counted prior precision
        total_prec = prec - (M - 1) / self.prior_var
        total_prec = np.maximum(total_prec, 1e-12)
        mean = mean_acc / total_prec
        if return_std:
            return mean, np.sqrt(1.0 / total_prec)
        return mean


__all__ = ["BayesianCommitteeMachine"]
