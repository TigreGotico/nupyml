"""Probabilistic PCA: the generative latent model behind PCA."""
import numpy as np

from ..base import BaseEstimator, TransformerMixin
from ..utils import check_array


class ProbabilisticPCA(BaseEstimator, TransformerMixin):
    """PCA as a probability MODEL, with a noise variance (Tipping & Bishop, 1999).

    Ordinary PCA is a projection with no notion of noise or likelihood.
    Probabilistic PCA writes the same subspace as a generative model:
    ``x = W z + mu + eps`` with a standard-normal latent ``z`` and ISOTROPIC
    Gaussian noise ``eps ~ N(0, sigma^2 I)``. Fitting it (closed form from the
    sample covariance's eigendecomposition) recovers not just the principal
    subspace but the leftover noise variance ``sigma^2`` -- the average of the
    discarded eigenvalues -- which turns PCA into something you can score with a
    proper likelihood, sample from, and use for missing-data or model comparison.
    As ``sigma^2 -> 0`` it reduces to classical PCA.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = check_array(X)
        n, d = X.shape
        q = self.n_components
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # eigendecomposition of the sample covariance
        S = (Xc.T @ Xc) / n
        vals, vecs = np.linalg.eigh(S)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        # the noise variance IS the mean of the discarded eigenvalues
        self.noise_variance_ = float(vals[q:].mean()) if d > q else 0.0
        Uq, Lq = vecs[:, :q], vals[:q]
        self.components_ = Uq.T
        self.explained_variance_ = Lq
        # W = U_q (Lambda_q - sigma^2 I)^{1/2}
        self.W_ = Uq * np.sqrt(np.maximum(Lq - self.noise_variance_, 0.0))
        self.M_ = self.W_.T @ self.W_ + self.noise_variance_ * np.eye(q)
        return self

    def transform(self, X):
        Xc = check_array(X) - self.mean_
        # posterior mean of the latent: M^{-1} W^T (x - mu)
        return Xc @ self.W_ @ np.linalg.inv(self.M_).T

    def inverse_transform(self, Z):
        return np.asarray(Z) @ self.W_.T + self.mean_

    def score_samples(self, X):
        Xc = check_array(X) - self.mean_
        d = Xc.shape[1]
        C = self.W_ @ self.W_.T + self.noise_variance_ * np.eye(d)
        sign, logdet = np.linalg.slogdet(C)
        Cinv = np.linalg.inv(C)
        maha = np.einsum("ij,jk,ik->i", Xc, Cinv, Xc)
        return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)

    def score(self, X, y=None):
        return float(self.score_samples(X).mean())


__all__ = ["ProbabilisticPCA"]
