"""Low-rank matrix recovery: robust PCA and matrix completion.

Both exploit the same prior -- the signal is LOW-RANK -- to recover it from
corruption (robust PCA) or from missing entries (matrix completion). The workhorse
is singular-value thresholding, the proximal operator of the nuclear norm.
"""
import numpy as np

from ..base import BaseEstimator, check_is_fitted
from ..utils import check_array


def _soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def singular_value_threshold(X, tau):
    """Proximal operator of the NUCLEAR norm: soft-threshold the singular values.

    Just as soft-thresholding the entries promotes SPARSITY (L1), soft-thresholding
    the SINGULAR VALUES promotes LOW RANK (nuclear norm) -- it shrinks every
    singular value by ``tau`` and drops those that hit zero, so small components
    vanish. This one operation is the engine of both algorithms below.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_t = np.maximum(s - tau, 0.0)
    return (U * s_t) @ Vt


class RobustPCA(BaseEstimator):
    """Split a matrix into LOW-RANK + SPARSE parts (Candès et al., 2011).

    THE PROBLEM WITH ORDINARY PCA
    -----------------------------
    PCA finds the best low-rank approximation in the L2 sense, so a FEW grossly
    corrupted entries (occlusions, spikes, sensor errors) wreck the whole fit --
    L2 has no tolerance for outliers. Robust PCA instead decomposes ``M = L + S``:
    a LOW-RANK ``L`` (the true structure) plus a SPARSE ``S`` (the arbitrary-
    magnitude corruption), by minimising ``||L||_* + lambda ||S||_1``. Astonishingly
    this convex program recovers BOTH exactly under broad conditions, so it cleanly
    separates a video's static background (low-rank) from its moving foreground
    (sparse), or clean data from gross errors.

    Solved by the inexact augmented-Lagrange-multiplier method: alternate an SVT
    step for ``L`` and a soft-threshold step for ``S``, updating a dual variable.
    """

    def __init__(self, lam=None, mu=None, max_iter=200, tol=1e-7):
        self.lam = lam
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, M):
        M = check_array(M)
        m, n = M.shape
        lam = self.lam or 1.0 / np.sqrt(max(m, n))
        norm2 = np.linalg.svd(M, compute_uv=False)[0]
        mu = self.mu or 1.25 / (norm2 + 1e-12)
        L = np.zeros_like(M)
        S = np.zeros_like(M)
        Y = M / max(norm2, np.abs(M).max() / lam)
        normM = np.linalg.norm(M, "fro")
        for _ in range(self.max_iter):
            L = singular_value_threshold(M - S + Y / mu, 1.0 / mu)   # low-rank step
            S = _soft_threshold(M - L + Y / mu, lam / mu)            # sparse step
            resid = M - L - S
            Y = Y + mu * resid                                       # dual update
            if np.linalg.norm(resid, "fro") / (normM + 1e-12) < self.tol:
                break
        self.low_rank_ = L
        self.sparse_ = S
        return self


class SoftImpute(BaseEstimator):
    """Matrix completion by iterative soft-thresholded SVD (Mazumder et al., 2010).

    Given a matrix with MISSING entries and the prior that the full matrix is
    low-rank, SoftImpute alternates: (1) fill the missing entries with the current
    low-rank estimate, (2) re-estimate by SVT of the filled matrix. Each step both
    fits the observed entries and shrinks the rank (the nuclear-norm penalty), and
    the iteration converges to the nuclear-norm-regularised completion. This is the
    scalable recommender-style completion behind "you might also like". Missing
    entries are marked ``NaN`` in the input.
    """

    def __init__(self, lam=1.0, max_iter=200, tol=1e-6):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    def fit_transform(self, M):
        M = np.array(M, dtype=float)
        observed = ~np.isnan(M)
        filled = np.where(observed, M, 0.0)
        X = filled.copy()
        for _ in range(self.max_iter):
            X_new = singular_value_threshold(X, self.lam)
            # keep observed entries, fill the rest from the low-rank estimate
            X_next = np.where(observed, M, X_new)
            if np.linalg.norm(X_next - X) / (np.linalg.norm(X) + 1e-12) < self.tol:
                X = X_next
                break
            X = X_next
        self.completed_ = X
        return X

    def fit(self, M):
        self.fit_transform(M)
        return self


__all__ = ["singular_value_threshold", "RobustPCA", "SoftImpute"]
