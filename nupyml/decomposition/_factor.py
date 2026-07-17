"""Factor analysis and incremental PCA: two more ways to find latent structure.

Both sit beside PCA in the decomposition module. Factor analysis differs from PCA
in what it assumes about the noise; incremental PCA differs in how it copes when
the data will not fit in memory.
"""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_array


class FactorAnalysis(BaseEstimator, TransformerMixin):
    """Latent factors plus PER-FEATURE noise -- PCA's probabilistic cousin.

    THE DIFFERENCE FROM PCA
    -----------------------
    PCA assumes the leftover variance is the same in every direction (isotropic
    noise), so a feature that is just noisy looks like a direction worth keeping.
    Factor analysis models the data as a few shared LATENT FACTORS plus INDEPENDENT
    per-feature noise::

        x = W z + noise,   noise_i ~ N(0, psi_i)   (a different psi per feature)

    Because each feature gets its own noise variance ``psi_i``, factor analysis
    can tell "this feature is intrinsically noisy" apart from "this feature loads
    on a real factor". PCA cannot -- it would rotate a high-variance noise feature
    into a top component. So when features have genuinely different noise levels,
    factor analysis recovers the true latent structure and PCA is misled.

    Fitted by EM: alternate estimating the latent factors given the parameters
    (E-step) and the loadings and noise given the factors (M-step).
    """

    def __init__(self, n_components=2, max_iter=100, tol=1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        X = check_array(X)
        n, p = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / n
        var = np.diag(cov).copy()

        k = self.n_components
        # init loadings from the top PCA directions; psi from residual variance.
        # the loadings are the RIGHT singular vectors (one row per feature),
        # scaled by the singular values -- shape (p, k)
        _, s, Vt = np.linalg.svd(Xc / np.sqrt(n), full_matrices=False)
        W = Vt[:k].T * s[:k]
        psi = np.maximum(var - (W ** 2).sum(axis=1), 1e-6)

        old_ll = -np.inf
        for _ in range(self.max_iter):
            # E-step: posterior over factors given current W, psi
            psi_inv = 1.0 / psi
            M = np.eye(k) + (W.T * psi_inv) @ W
            M_inv = np.linalg.inv(M)
            beta = M_inv @ (W.T * psi_inv)             # factor posterior mean map
            Ez = Xc @ beta.T                            # (n, k)
            Ezz = n * M_inv + Ez.T @ Ez                 # posterior second moment
            # M-step: re-estimate loadings and per-feature noise
            W = (Xc.T @ Ez) @ np.linalg.inv(Ezz)
            psi = np.maximum(var - np.sum(W * (Xc.T @ Ez) / n, axis=1), 1e-6)

            ll = -0.5 * n * (np.sum(np.log(psi)) + np.log(np.linalg.det(M)))
            if abs(ll - old_ll) < self.tol:
                break
            old_ll = ll

        self.components_ = W.T
        self.noise_variance_ = psi
        self.loglik_ = ll
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        Xc = check_array(X) - self.mean_
        W = self.components_.T
        psi_inv = 1.0 / self.noise_variance_
        M = np.eye(self.n_components) + (W.T * psi_inv) @ W
        return Xc @ (np.linalg.inv(M) @ (W.T * psi_inv)).T


class IncrementalPCA(BaseEstimator, TransformerMixin):
    """PCA for data that will not fit in memory -- fit in batches.

    THE PROBLEM
    -----------
    Ordinary PCA needs the whole covariance matrix, which needs the whole dataset
    at once. For data too large for memory (or arriving as a stream),
    ``partial_fit`` updates the components from one BATCH at a time, keeping only a
    running summary -- never the full data.

    THE MECHANISM
    -------------
    Merge each batch into the current low-rank estimate: stack the existing
    components (weighted by their singular values) with the new batch, and re-SVD
    that small combined matrix. The cost per batch depends on the number of
    components, not on how much data has already been seen -- so memory stays
    bounded no matter how long the stream. The result approximates full PCA
    closely; the approximation is in how the batches are merged, and it is the
    standard way to PCA a dataset larger than RAM.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def partial_fit(self, X, y=None):
        X = check_array(X)
        k = self.n_components
        if not hasattr(self, "components_"):
            self.mean_ = X.mean(axis=0)
            self.n_samples_seen_ = len(X)
            Xc = X - self.mean_
            U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            self.components_ = Vt[:k]
            self.singular_values_ = s[:k]
            return self

        # update the running mean, then merge old components with the new batch
        n_old = self.n_samples_seen_
        n_new = len(X)
        old_mean = self.mean_
        new_mean = X.mean(axis=0)
        total = n_old + n_new
        self.mean_ = (n_old * old_mean + n_new * new_mean) / total

        Xc = X - self.mean_
        # reconstruct the old data's variance from stored components, and a
        # mean-shift correction term, then re-SVD the small stacked matrix
        mean_correction = np.sqrt(n_old * n_new / total) * (old_mean - new_mean)
        combined = np.vstack([
            self.singular_values_[:, None] * self.components_,
            Xc,
            mean_correction[None]])
        U, s, Vt = np.linalg.svd(combined, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = s[:self.n_components]
        self.n_samples_seen_ = total
        return self

    def fit(self, X, y=None, batch_size=None):
        X = check_array(X)
        batch_size = batch_size or max(self.n_components + 1, len(X) // 5)
        for start in range(0, len(X), batch_size):
            self.partial_fit(X[start:start + batch_size])
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        return (check_array(X) - self.mean_) @ self.components_.T


__all__ = ["FactorAnalysis", "IncrementalPCA"]
