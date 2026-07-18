"""The hidden semi-Markov model: an HMM whose states have EXPLICIT durations.

In a plain HMM the time spent in a state is geometric -- the probability of
staying falls off by a constant factor each step, so short stays are always the
most likely, no matter the process. That is wrong for anything with a
characteristic duration: a spoken phoneme, a sleep stage, a machine's operating
mode. An HSMM fixes it by giving each state its OWN duration distribution: the
model picks a state, draws how long to stay from that state's duration law, emits
that many observations, then transitions (never to itself). This file implements
generation, segmental Viterbi decoding, and a hard-EM (segmental k-means) fit.
"""
import numpy as np

from ..base import BaseEstimator, check_is_fitted
from ..utils import check_array, check_random_state


class HiddenSemiMarkovModel(BaseEstimator):
    """HMM with per-state explicit-duration distributions and Gaussian emissions.

    ``max_duration`` caps how long a single visit to a state can last (the support
    of each state's duration distribution). Emissions are diagonal-covariance
    Gaussians, like ``GaussianHMM``. The transition matrix has a zero diagonal --
    duration is modelled explicitly, so a "self-transition" would double-count it.
    """

    def __init__(self, n_states=2, max_duration=20, max_iter=20, tol=1e-3,
                 random_state=None):
        self.n_states = n_states
        self.max_duration = max_duration
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # ---- emission log-likelihoods -------------------------------------------
    def _log_emit(self, X):
        diff = X[:, None, :] - self.means_[None]         # (T, K, D)
        var = self.covars_[None]
        return -0.5 * (np.log(2 * np.pi * var) + diff ** 2 / var).sum(axis=-1)

    # ---- segmental Viterbi ---------------------------------------------------
    def decode(self, X):
        """Most likely (state, duration) segmentation via explicit-duration Viterbi."""
        check_is_fitted(self, "transmat_")
        X = check_array(X)
        T, K, Dmax = len(X), self.n_states, self.max_duration
        log_e = self._log_emit(X)                        # (T, K)
        cum = np.vstack([np.zeros(K), np.cumsum(log_e, axis=0)])  # prefix sums
        log_A = np.log(self.transmat_ + 1e-300)
        log_dur = np.log(self.durations_ + 1e-300)       # (K, Dmax)
        log_start = np.log(self.startprob_ + 1e-300)
        NEG = -1e300
        delta = np.full((T + 1, K), NEG)
        back = np.zeros((T + 1, K, 2), dtype=int)        # (prev_state, duration)
        delta[0] = 0.0
        for t in range(1, T + 1):
            for d in range(1, min(Dmax, t) + 1):
                seg_ll = cum[t] - cum[t - d]             # emission ll of segment
                for j in range(K):
                    dur_ll = log_dur[j, d - 1]
                    if t - d == 0:
                        cand = log_start[j] + dur_ll + seg_ll[j]
                        if cand > delta[t, j]:
                            delta[t, j] = cand; back[t, j] = (-1, d)
                    else:
                        prev = delta[t - d] + log_A[:, j]
                        i = int(prev.argmax())
                        cand = prev[i] + dur_ll + seg_ll[j]
                        if cand > delta[t, j]:
                            delta[t, j] = cand; back[t, j] = (i, d)
        states = np.empty(T, dtype=int)
        t, j = T, int(delta[T].argmax())
        while t > 0:
            i, d = back[t, j]
            states[t - d:t] = j
            t, j = t - d, i
        return states

    # ---- hard-EM fit (segmental k-means) ------------------------------------
    def fit(self, X):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        T, D = X.shape
        K = self.n_states
        # init emissions from k-means-style seeds
        idx = rng.choice(T, size=K, replace=False)
        self.means_ = X[idx].astype(float)
        self.covars_ = np.tile(X.var(axis=0) + 1e-2, (K, 1))
        self.startprob_ = np.full(K, 1.0 / K)
        A = np.ones((K, K)) - np.eye(K)
        self.transmat_ = A / A.sum(axis=1, keepdims=True)
        self.durations_ = np.full((K, self.max_duration), 1.0 / self.max_duration)
        prev = None
        for it in range(self.max_iter):
            states = self.decode(X)
            self._reestimate(X, states)
            key = states.tobytes()
            if key == prev:                              # converged (stable labels)
                break
            prev = key
        self.n_iter_ = it + 1
        return self

    def _reestimate(self, X, states):
        K, Dmax = self.n_states, self.max_duration
        # segments = maximal runs of a single state
        segs = []
        s = 0
        for t in range(1, len(states) + 1):
            if t == len(states) or states[t] != states[s]:
                segs.append((states[s], s, t)); s = t
        dur_counts = np.ones((K, Dmax))                  # Laplace smoothing
        trans_counts = np.ones((K, K)) - np.eye(K) + 1e-9
        start_counts = np.ones(K) * 1e-9
        start_counts[segs[0][0]] += 1
        for k, (st, a, b) in enumerate(segs):
            d = min(b - a, Dmax)
            dur_counts[st, d - 1] += 1
            if k + 1 < len(segs):
                trans_counts[st, segs[k + 1][0]] += 1
        self.durations_ = dur_counts / dur_counts.sum(axis=1, keepdims=True)
        self.transmat_ = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        self.startprob_ = start_counts / start_counts.sum()
        for j in range(K):
            pts = X[states == j]
            if len(pts):
                self.means_[j] = pts.mean(axis=0)
                self.covars_[j] = np.maximum(pts.var(axis=0), 1e-3)

    # ---- generation ---------------------------------------------------------
    def sample(self, n_samples, random_state=None):
        check_is_fitted(self, "transmat_")
        rng = check_random_state(random_state)
        K, D = self.means_.shape
        X, states = [], []
        j = rng.choice(K, p=self.startprob_)
        while len(X) < n_samples:
            d = rng.choice(self.max_duration, p=self.durations_[j]) + 1
            for _ in range(d):
                if len(X) >= n_samples:
                    break
                X.append(rng.normal(self.means_[j], np.sqrt(self.covars_[j])))
                states.append(j)
            j = rng.choice(K, p=self.transmat_[j])
        return np.array(X), np.array(states)


__all__ = ["HiddenSemiMarkovModel"]
