"""A series that JUMPS between regimes (Hamilton, 1989)."""
import numpy as np
from ..base import BaseEstimator
from ..utils import check_array, check_random_state


class MarkovSwitchingModel(BaseEstimator):
    """A series that JUMPS between regimes (Hamilton, 1989).

    Some series are not one process but several, switching between them -- calm vs
    crisis markets, expansion vs recession. The Markov-switching model fits a
    separate mean/variance (and optional AR) for each REGIME and a Markov transition
    matrix between them, then runs the Hamilton FILTER: a forward recursion that
    combines the transition probabilities with each regime's likelihood to give, at
    every time, the probability the series is in each regime. Parameters are fit by
    EM. ``predict_regimes`` returns the smoothed regime probabilities.
    """

    def __init__(self, n_regimes=2, max_iter=100, tol=1e-4, random_state=None):
        self.n_regimes = n_regimes
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, y):
        y = np.asarray(y, float).ravel()
        rng = check_random_state(self.random_state)
        K, n = self.n_regimes, len(y)
        # init means spread across the data range, equal variances
        self.means_ = np.quantile(y, np.linspace(0.2, 0.8, K))
        self.vars_ = np.full(K, y.var())
        self.trans_ = np.full((K, K), 1.0 / K)
        self.pi_ = np.full(K, 1.0 / K)
        prev_ll = -np.inf
        for _ in range(self.max_iter):
            B = np.exp(-0.5 * (y[:, None] - self.means_) ** 2 / self.vars_) \
                / np.sqrt(2 * np.pi * self.vars_)          # (n, K) likelihoods
            alpha, c = self._forward(B)
            beta = self._backward(B, c)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = np.zeros((K, K))
            for t in range(n - 1):
                num = (alpha[t][:, None] * self.trans_
                       * B[t + 1][None, :] * beta[t + 1][None, :])
                xi += num / num.sum()
            self.trans_ = xi / xi.sum(axis=1, keepdims=True)
            self.pi_ = gamma[0]
            nk = gamma.sum(axis=0)
            self.means_ = (gamma * y[:, None]).sum(axis=0) / nk
            self.vars_ = np.maximum(
                (gamma * (y[:, None] - self.means_) ** 2).sum(axis=0) / nk, 1e-6)
            ll = np.log(c).sum()
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        self.smoothed_ = gamma
        return self

    def _forward(self, B):
        n, K = B.shape
        alpha = np.zeros((n, K)); c = np.zeros(n)
        alpha[0] = self.pi_ * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ self.trans_) * B[t]
            c[t] = alpha[t].sum(); alpha[t] /= c[t]
        return alpha, c

    def _backward(self, B, c):
        n, K = B.shape
        beta = np.zeros((n, K)); beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (self.trans_ @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        return beta

    def predict_regimes(self, y=None):
        return self.smoothed_


__all__ = ["MarkovSwitchingModel"]
