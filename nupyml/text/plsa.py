"""Probabilistic latent semantic analysis: topics as a proper factorisation."""
import numpy as np

from ..base import BaseEstimator
from ..utils import check_random_state


class PLSA(BaseEstimator):
    """Topics as a PROBABILISTIC factorisation of the term matrix (Hofmann, 1999).

    Latent semantic analysis factorises the document-term matrix with an SVD, whose
    negative entries have no meaning. pLSA replaces it with a proper generative model:
    each document is a mixture over TOPICS, each topic a distribution over WORDS, and
    every count is explained by summing over the hidden topic that generated it. EM
    alternates between guessing which topic produced each word (E-step) and re-
    estimating the topic-word and document-topic distributions (M-step). It is LDA
    without the Dirichlet prior, and the clearest way to see topic models as
    factorisation. ``n_topics`` sets the number of latent topics.
    """

    def __init__(self, n_topics=10, max_iter=50, tol=1e-4, random_state=None):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, float)
        rng = check_random_state(self.random_state)
        D, W = X.shape
        K = self.n_topics
        p_wz = rng.rand(K, W); p_wz /= p_wz.sum(axis=1, keepdims=True)   # p(w|z)
        p_zd = rng.rand(D, K); p_zd /= p_zd.sum(axis=1, keepdims=True)   # p(z|d)
        prev = -np.inf
        for _ in range(self.max_iter):
            # E-step responsibilities p(z|d,w) folded straight into the M-step counts
            new_wz = np.zeros((K, W)); new_zd = np.zeros((D, K))
            for d in range(D):
                nz = np.nonzero(X[d])[0]
                if len(nz) == 0:
                    continue
                joint = p_zd[d][:, None] * p_wz[:, nz]      # (K, |nz|)
                resp = joint / (joint.sum(axis=0, keepdims=True) + 1e-12)
                counts = X[d, nz]
                new_wz[:, nz] += resp * counts
                new_zd[d] += (resp * counts).sum(axis=1)
            p_wz = new_wz / (new_wz.sum(axis=1, keepdims=True) + 1e-12)
            p_zd = new_zd / (new_zd.sum(axis=1, keepdims=True) + 1e-12)
            ll = self._loglik(X, p_wz, p_zd)
            if abs(ll - prev) < self.tol * abs(prev + 1e-12):
                break
            prev = ll
        self.topic_word_ = p_wz
        self.doc_topic_ = p_zd
        self.loglik_ = prev
        return self

    @staticmethod
    def _loglik(X, p_wz, p_zd):
        pdw = p_zd @ p_wz                                   # p(w|d)
        return (X * np.log(pdw + 1e-12)).sum()

    def transform(self, X):
        return self.fit(X).doc_topic_


__all__ = ["PLSA"]
