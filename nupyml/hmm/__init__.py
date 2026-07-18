"""Hidden Markov models: sequences with unobserved state.

THE MODEL
---------
A system moves between STATES you cannot see, and at each step emits an
observation you can. Speech: the state is the phoneme, the observation is the
audio. The Markov assumption is that the next state depends only on the current
one -- the past is summarised entirely by where you are now.

Three parameters: ``startprob_`` (where it begins), ``transmat_`` (how it moves),
and the emissions (what each state produces).

THREE QUESTIONS, THREE ALGORITHMS
---------------------------------
* *How likely is this sequence?* -- the FORWARD algorithm (``score``).
  Summing over every possible state path is exponential; dynamic programming
  makes it linear, because all paths through a state at time t share a future.
  ``_forward`` accumulates that shared work once.
* *What states did it pass through?* -- VITERBI (``predict``). Identical
  structure, with max replacing sum: instead of the total probability of
  reaching a state, keep the best single path to it, and backtrack at the end.
  Note this returns the best PATH, which is not the same as the sequence of
  individually-best states -- and only Viterbi's answer is guaranteed to be a
  legal path.
* *What are the parameters?* -- BAUM-WELCH (``fit``), which is EM for HMMs. The
  E-step computes state posteriors with forward-backward, the M-step re-estimates
  the parameters from them. Same circularity, same fix, same local optimum as the
  GMM in ``nupyml.mixture``.

WHY EVERYTHING IS IN LOG SPACE
------------------------------
The probability of a specific 1000-step path is a product of 1000 numbers below
1: it underflows to exactly zero long before the sequence ends, and then every
ratio is 0/0. Logs turn the products into sums, and ``logsumexp`` adds
probabilities in log space without ever leaving it.
"""
import numpy as np
from scipy.special import logsumexp

from ..base import BaseEstimator, check_is_fitted
from ..utils import check_random_state
from ._hsmm import HiddenSemiMarkovModel


class _BaseHMM(BaseEstimator):
    def __init__(self, n_components=1, max_iter=100, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # subclasses provide _log_likelihood_matrix(X) -> (T, K) and
    # _m_step_emissions(X, gamma)

    def _forward(self, log_b):
        T, K = log_b.shape
        log_alpha = np.empty((T, K))
        log_alpha[0] = np.log(self.startprob_) + log_b[0]
        log_A = np.log(self.transmat_)
        for t in range(1, T):
            log_alpha[t] = log_b[t] + logsumexp(
                log_alpha[t - 1][:, None] + log_A, axis=0)
        return log_alpha

    def _backward(self, log_b):
        T, K = log_b.shape
        log_beta = np.zeros((T, K))
        log_A = np.log(self.transmat_)
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_A + log_b[t + 1] + log_beta[t + 1], axis=1)
        return log_beta

    def score(self, X):
        check_is_fitted(self, "transmat_")
        log_b = self._log_likelihood_matrix(np.asarray(X))
        return float(logsumexp(self._forward(log_b)[-1]))

    def predict(self, X):
        """Viterbi decoding of the most likely state sequence."""
        check_is_fitted(self, "transmat_")
        X = np.asarray(X)
        log_b = self._log_likelihood_matrix(X)
        T, K = log_b.shape
        log_A = np.log(self.transmat_)
        delta = np.log(self.startprob_) + log_b[0]
        psi = np.zeros((T, K), dtype=int)
        for t in range(1, T):
            scores = delta[:, None] + log_A
            psi[t] = scores.argmax(axis=0)
            delta = scores.max(axis=0) + log_b[t]
        states = np.empty(T, dtype=int)
        states[-1] = delta.argmax()
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def predict_proba(self, X):
        check_is_fitted(self, "transmat_")
        log_b = self._log_likelihood_matrix(np.asarray(X))
        log_alpha = self._forward(log_b)
        log_beta = self._backward(log_b)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def fit(self, X, lengths=None):
        X = np.asarray(X)
        sequences = self._split(X, lengths)
        rng = check_random_state(self.random_state)
        K = self.n_components
        self.startprob_ = np.full(K, 1.0 / K)
        A = rng.uniform(0.5, 1.5, size=(K, K))
        self.transmat_ = A / A.sum(axis=1, keepdims=True)
        self._init_emissions(X, rng)
        prev_ll = -np.inf
        for it in range(self.max_iter):
            ll_total = 0.0
            start_num = np.zeros(K)
            trans_num = np.zeros((K, K))
            gammas = []
            for seq in sequences:
                log_b = self._log_likelihood_matrix(seq)
                log_alpha = self._forward(log_b)
                log_beta = self._backward(log_b)
                ll = logsumexp(log_alpha[-1])
                ll_total += ll
                log_gamma = log_alpha + log_beta - ll
                gamma = np.exp(log_gamma)
                gammas.append(gamma)
                start_num += gamma[0]
                if len(seq) > 1:
                    log_A = np.log(self.transmat_)
                    log_xi = (log_alpha[:-1, :, None] + log_A[None]
                              + log_b[1:, None, :] + log_beta[1:, None, :] - ll)
                    trans_num += np.exp(logsumexp(log_xi, axis=0))
            self.startprob_ = np.maximum(start_num / len(sequences), 1e-12)
            self.startprob_ /= self.startprob_.sum()
            trans_num = np.maximum(trans_num, 1e-12)
            self.transmat_ = trans_num / trans_num.sum(axis=1, keepdims=True)
            self._m_step_emissions(sequences, gammas)
            if abs(ll_total - prev_ll) < self.tol:
                break
            prev_ll = ll_total
        self.n_iter_ = it + 1
        self.log_likelihood_ = ll_total
        return self

    @staticmethod
    def _split(X, lengths):
        if lengths is None:
            return [X]
        out = []
        start = 0
        for L in lengths:
            out.append(X[start:start + L])
            start += L
        return out


class GaussianHMM(_BaseHMM):
    """HMM with diagonal-covariance Gaussian emissions. X: (T, D)."""

    def _init_emissions(self, X, rng):
        K = self.n_components
        idx = rng.choice(len(X), size=K, replace=False)
        self.means_ = X[idx].astype(np.float64)
        self.covars_ = np.tile(X.var(axis=0) + 1e-3, (K, 1))

    def _log_likelihood_matrix(self, X):
        diff = X[:, None, :] - self.means_[None, :, :]     # (T, K, D)
        var = self.covars_[None]
        return -0.5 * (np.log(2 * np.pi * var) + diff ** 2 / var).sum(axis=-1)

    def _m_step_emissions(self, sequences, gammas):
        X = np.vstack(sequences)
        G = np.vstack(gammas)                              # (T_total, K)
        nk = G.sum(axis=0) + 1e-10
        self.means_ = (G.T @ X) / nk[:, None]
        sq = (G.T @ (X ** 2)) / nk[:, None] - self.means_ ** 2
        self.covars_ = np.maximum(sq, 1e-6)

    def sample(self, n_samples, random_state=None):
        check_is_fitted(self, "transmat_")
        rng = check_random_state(random_state)
        K, D = self.means_.shape
        states = np.empty(n_samples, dtype=int)
        X = np.empty((n_samples, D))
        states[0] = rng.choice(K, p=self.startprob_)
        for t in range(1, n_samples):
            states[t] = rng.choice(K, p=self.transmat_[states[t - 1]])
        for t in range(n_samples):
            X[t] = rng.normal(self.means_[states[t]],
                              np.sqrt(self.covars_[states[t]]))
        return X, states


class MultinomialHMM(_BaseHMM):
    """HMM with categorical emissions. X: (T,) integer symbols."""

    def __init__(self, n_components=1, n_symbols=None, max_iter=100, tol=1e-4,
                 random_state=None):
        super().__init__(n_components, max_iter, tol, random_state)
        self.n_symbols = n_symbols

    def _init_emissions(self, X, rng):
        V = self.n_symbols or int(X.max()) + 1
        self._V = V
        B = rng.uniform(0.5, 1.5, size=(self.n_components, V))
        self.emissionprob_ = B / B.sum(axis=1, keepdims=True)

    def _log_likelihood_matrix(self, X):
        return np.log(self.emissionprob_[:, X.astype(int)]).T

    def _m_step_emissions(self, sequences, gammas):
        X = np.concatenate(sequences).astype(int)
        G = np.vstack(gammas)
        B = np.zeros((self.n_components, self._V))
        for v in range(self._V):
            B[:, v] = G[X == v].sum(axis=0)
        B = np.maximum(B, 1e-12)
        self.emissionprob_ = B / B.sum(axis=1, keepdims=True)

    def sample(self, n_samples, random_state=None):
        check_is_fitted(self, "transmat_")
        rng = check_random_state(random_state)
        states = np.empty(n_samples, dtype=int)
        X = np.empty(n_samples, dtype=int)
        states[0] = rng.choice(self.n_components, p=self.startprob_)
        X[0] = rng.choice(self._V, p=self.emissionprob_[states[0]])
        for t in range(1, n_samples):
            states[t] = rng.choice(self.n_components, p=self.transmat_[states[t - 1]])
            X[t] = rng.choice(self._V, p=self.emissionprob_[states[t]])
        return X, states


__all__ = ["GaussianHMM", "MultinomialHMM", "HiddenSemiMarkovModel"]
