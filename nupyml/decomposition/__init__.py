"""Matrix decomposition: rewriting X as a product of simpler factors.

Every method here factors ``X ~ W @ H`` -- each row of X becomes a combination
of a few shared components. They differ ONLY in what they demand of the factors,
and each demand buys a different kind of interpretability:

=================== ==================== =====================================
method              constraint           what you get
=================== ==================== =====================================
``PCA``             orthogonal           uncorrelated directions of max variance
``TruncatedSVD``    orthogonal, no       same, but works on sparse data
                    centring
``NMF``             non-negative         PARTS, because nothing can cancel
``FastICA``         statistically        independent SOURCES, not just
                    independent          uncorrelated ones
``SparsePCA``       few non-zeros        components you can actually read
``DictionaryLearning``  sparse codes,    an overcomplete basis
                    overcomplete
=================== ==================== =====================================

PCA VS ICA: UNCORRELATED IS NOT INDEPENDENT
-------------------------------------------
PCA finds directions of maximum variance, and its components are uncorrelated --
which only constrains SECOND-order statistics. Independence is much stronger.
Given two people talking over each other, PCA gives the two loudest orthogonal
directions; ICA gives the two voices. That is why ICA is the tool for blind
source separation and PCA is not.

WHY NMF GIVES PARTS
-------------------
With signs allowed, a component can cancel another, and PCA components typically
do exactly that: "this pattern, MINUS that one". Useful, and hard to interpret --
what is negative brightness?

Forbid negatives and cancellation becomes impossible, so the only way to build
data is by ADDING pieces. Trained on faces, NMF produces noses and eyebrows
where PCA produces ghostly whole-face templates. The constraint IS the
interpretability, and it costs you: the problem is no longer convex, so there is
no unique answer and the initialisation matters.

KERNEL PCA
----------
PCA in a feature space you never construct. The kernel trick: PCA can be written
purely in terms of inner products, so replacing them with ``k(x, x')`` performs
PCA in whatever (possibly infinite-dimensional) space that kernel corresponds
to. Non-linear structure becomes linear there.

CHOOSING THE NUMBER OF COMPONENTS
---------------------------------
``explained_variance_ratio_`` is the usual guide -- and note that PCA maximises
variance, which is not the same as usefulness. A low-variance direction can
carry the entire signal you care about, and PCA will discard it without
hesitation, because it never looks at the labels. ``LinearDiscriminantAnalysis``
does.
"""
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.spatial.distance import cdist

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


def _deterministic_signs(components):
    """Fix the arbitrary sign of each component.

    An eigenvector is only defined up to sign: if ``v`` is one, so is ``-v``.
    Solvers are free to return either, which makes results annoyingly
    irreproducible across methods. Convention: force the entry of largest
    magnitude in each component to be positive.
    """
    peak = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(len(components)), peak])
    signs[signs == 0] = 1.0
    return components * signs[:, None]


def _randomized_svd(X, k, n_oversamples=10, n_iter=4, rng=None):
    """Approximate the top ``k`` singular triplets (Halko, Martinsson & Tropp).

    An exact SVD costs O(n*d*min(n,d)) -- wasteful when you only want a handful
    of components. The randomized method works in two stages:

    1. *Find the subspace.* Multiply X by a random matrix with ``k + p``
       columns. Each product is a random mixture of X's columns, so it lands
       (mostly) inside the span of X's leading singular directions -- the
       strong directions dominate any random mixture. That gives a thin basis Q
       capturing nearly all of X's action.
    2. *Solve the small problem.* Project X onto Q, giving a tiny
       ``(k+p) x d`` matrix, and SVD that exactly. Map the result back.

    ``n_oversamples`` grabs a few extra directions so that noise in the random
    draw does not cost us a real component. ``n_iter`` power iterations
    (multiplying by ``X X'`` a few times) push the weaker singular values down
    relative to the strong ones, sharpening the subspace when the spectrum
    decays slowly.
    """
    rng = check_random_state(rng)
    n, d = X.shape
    size = min(k + n_oversamples, d)
    Q = rng.normal(size=(d, size))
    Q, _ = np.linalg.qr(X @ Q)
    for _ in range(n_iter):
        # power iteration, re-orthonormalised each step to fight round-off
        Q, _ = np.linalg.qr(X.T @ Q)
        Q, _ = np.linalg.qr(X @ Q)
    B = Q.T @ X                       # small: (size, d)
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ Ub)[:, :k], S[:k], Vt[:k]


class PCA(BaseEstimator, TransformerMixin):
    """Principal component analysis.

    PCA finds the orthogonal directions along which the data varies most. Every
    route to them is a different way of diagonalising the same covariance, and
    they differ only in cost and numerical care:

    ``"full"``
        SVD of the centred data. The most accurate, and the reference the
        others are judged against. Costs O(n*d*min(n, d)).

    ``"covariance_eigh"``
        When there are far more samples than features, forming the ``d x d``
        covariance ``X'X`` and eigendecomposing *that* is much cheaper: the
        expensive part becomes O(n*d^2) with a tiny O(d^3) tail, and none of it
        touches an n-by-d factorisation.

        The catch is precision. Squaring the data squares the condition number,
        so singular values below roughly ``sqrt(eps)`` times the largest are
        lost in round-off. For well-conditioned data this is invisible; for
        nearly-collinear features it is not. That is the trade being made, and
        why it is not the universal default.

    ``"randomized"``
        Only the leading ``k`` components, via random projection. Wins when
        ``k`` is much smaller than the data's dimensions.

    ``"auto"``
        Picks ``covariance_eigh`` when samples greatly outnumber features (the
        case where it is both a large win and numerically comfortable),
        ``randomized`` when only a few components of a large matrix are wanted,
        and ``full`` otherwise.
    """

    def __init__(self, n_components=None, whiten=False, svd_solver="auto",
                 random_state=None):
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.random_state = random_state

    def _choose_solver(self, n, d, k):
        if self.svd_solver != "auto":
            return self.svd_solver
        if n >= 10 * d and d <= 1000:
            return "covariance_eigh"
        if k < 0.8 * min(n, d) and max(n, d) > 500:
            return "randomized"
        return "full"

    def fit(self, X, y=None):
        X = check_array(X)
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # total variance is the same however we factorise: it is just the
        # squared Frobenius norm, so a truncated solver can still report ratios
        total_var = float((Xc ** 2).sum()) / (n - 1)

        k_req = self.n_components
        if k_req is None:
            k_req = min(n, d)
        solver = self._choose_solver(n, d, min(int(k_req) if not
                                               isinstance(k_req, float)
                                               else d, min(n, d)))

        if solver == "covariance_eigh":
            cov = Xc.T @ Xc
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(-eigvals)
            eigvals = np.maximum(eigvals[order], 0.0)
            components = eigvecs[:, order].T
            singular = np.sqrt(eigvals)
        elif solver == "randomized":
            k_fit = min(int(k_req) if not isinstance(k_req, float) else d,
                        min(n, d))
            _, singular, components = _randomized_svd(
                Xc, k_fit, rng=self.random_state)
        elif solver == "full":
            _, singular, components = np.linalg.svd(Xc, full_matrices=False)
        else:
            raise ValueError(f"Unknown svd_solver: {self.svd_solver!r}")

        explained = singular ** 2 / (n - 1)
        if isinstance(k_req, float):
            # keep the fewest components covering this fraction of variance
            ratios = explained / total_var
            k = int(np.searchsorted(np.cumsum(ratios), k_req) + 1)
        else:
            k = min(int(k_req), len(singular))

        self.components_ = _deterministic_signs(components[:k])
        self.singular_values_ = singular[:k]
        self.explained_variance_ = explained[:k]
        self.explained_variance_ratio_ = explained[:k] / total_var
        self.n_components_ = k
        self.n_features_in_ = d
        self.n_features_out_ = k
        self.svd_solver_ = solver
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X)
        Xt = (X - self.mean_) @ self.components_.T
        if self.whiten:
            Xt /= np.sqrt(self.explained_variance_)
        return Xt

    def inverse_transform(self, Xt):
        check_is_fitted(self, "components_")
        Xt = np.asarray(Xt, dtype=np.float64)
        if self.whiten:
            Xt = Xt * np.sqrt(self.explained_variance_)
        return Xt @ self.components_ + self.mean_


class TruncatedSVD(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        if sp.issparse(X):
            U, S, Vt = sp.linalg.svds(X, k=self.n_components)
            order = np.argsort(-S)
            U, S, Vt = U[:, order], S[order], Vt[order]
        else:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            U, S, Vt = U[:, :self.n_components], S[:self.n_components], \
                Vt[:self.n_components]
        self.components_ = Vt
        self.singular_values_ = S
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = len(Vt)
        return U * S

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X, accept_sparse=True)
        return np.asarray(X @ self.components_.T)


class NMF(BaseEstimator, TransformerMixin):
    """Non-negative matrix factorization via multiplicative updates."""

    _estimator_tags = {"requires_positive_X": True}

    def __init__(self, n_components=2, max_iter=500, tol=1e-5, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        X = check_array(X)
        if (X < 0).any():
            raise ValueError("NMF input must be non-negative")
        rng = check_random_state(self.random_state)
        n, d = X.shape
        k = self.n_components
        scale = np.sqrt(X.mean() / k)
        W = np.abs(rng.normal(scale=scale, size=(n, k)))
        H = np.abs(rng.normal(scale=scale, size=(k, d)))
        eps = 1e-12
        prev_err = np.inf
        for it in range(self.max_iter):
            H *= (W.T @ X) / (W.T @ W @ H + eps)
            W *= (X @ H.T) / (W @ H @ H.T + eps)
            if it % 10 == 0:
                err = np.linalg.norm(X - W @ H)
                if abs(prev_err - err) < self.tol * max(prev_err, 1.0):
                    break
                prev_err = err
        self.components_ = H
        self.reconstruction_err_ = float(np.linalg.norm(X - W @ H))
        self.n_iter_ = it + 1
        self.n_features_in_ = d
        self.n_features_out_ = k
        return W

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X)
        H = self.components_
        rng = check_random_state(self.random_state)
        W = np.abs(rng.normal(scale=np.sqrt(X.mean() / len(H)),
                              size=(len(X), len(H))))
        eps = 1e-12
        for _ in range(200):
            W *= (X @ H.T) / (W @ H @ H.T + eps)
        return W

    def inverse_transform(self, W):
        check_is_fitted(self, "components_")
        return np.asarray(W) @ self.components_


class FastICA(BaseEstimator, TransformerMixin):
    """FastICA with the logcosh contrast and symmetric decorrelation."""

    def __init__(self, n_components=None, max_iter=200, tol=1e-5,
                 random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n, d = X.shape
        k = self.n_components or d
        self.mean_ = X.mean(axis=0)
        Xc = (X - self.mean_).T                       # (d, n)
        # whiten
        cov = Xc @ Xc.T / n
        eigval, eigvec = np.linalg.eigh(cov)
        order = np.argsort(-eigval)[:k]
        D = eigval[order]
        E = eigvec[:, order]
        K = (E / np.sqrt(D)).T                        # (k, d)
        Xw = K @ Xc                                   # (k, n)
        W = rng.normal(size=(k, k))

        def sym_decorrelate(W):
            s, u = np.linalg.eigh(W @ W.T)
            return (u / np.sqrt(s)) @ u.T @ W

        W = sym_decorrelate(W)
        for _ in range(self.max_iter):
            WX = W @ Xw
            g = np.tanh(WX)
            g_prime = 1 - g ** 2
            W_new = g @ Xw.T / n - g_prime.mean(axis=1)[:, None] * W
            W_new = sym_decorrelate(W_new)
            lim = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1))
            W = W_new
            if lim < self.tol:
                break
        self.whitening_ = K
        self.unmixing_ = W
        self.components_ = W @ K
        self.mixing_ = np.linalg.pinv(self.components_)
        self.n_features_in_ = d
        self.n_features_out_ = k
        return (self.components_ @ Xc).T

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "components_")
        X = check_array(X)
        return (X - self.mean_) @ self.components_.T


class KernelPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, kernel="rbf", gamma=None, degree=3):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree

    def _kernel(self, X, Y):
        gamma = self.gamma or 1.0 / X.shape[1]
        if self.kernel == "rbf":
            return np.exp(-gamma * cdist(X, Y) ** 2)
        if self.kernel == "poly":
            return (gamma * X @ Y.T + 1.0) ** self.degree
        if self.kernel == "linear":
            return X @ Y.T
        raise ValueError(f"Unknown kernel: {self.kernel!r}")

    def fit(self, X, y=None):
        X = check_array(X)
        self._X_fit = X
        n = len(X)
        K = self._kernel(X, X)
        one_n = np.full((n, n), 1.0 / n)
        Kc = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        vals, vecs = scipy.linalg.eigh(Kc)
        order = np.argsort(-vals)[:self.n_components]
        self.eigenvalues_ = np.maximum(vals[order], 0)
        self.eigenvectors_ = vecs[:, order]
        self._K_fit_rows = K.mean(axis=0)
        self._K_fit_all = K.mean()
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = self.n_components
        return self

    def transform(self, X):
        check_is_fitted(self, "eigenvectors_")
        X = check_array(X)
        K = self._kernel(X, self._X_fit)
        Kc = (K - K.mean(axis=1, keepdims=True) - self._K_fit_rows
              + self._K_fit_all)
        nonzero = self.eigenvalues_ > 1e-12
        alphas = np.zeros_like(self.eigenvectors_)
        alphas[:, nonzero] = (self.eigenvectors_[:, nonzero]
                              / np.sqrt(self.eigenvalues_[nonzero]))
        return Kc @ alphas


from ._sparse import (  # noqa: E402
    SparsePCA, DictionaryLearning, SparseCoder, MiniBatchNMF,
    LatentDirichletAllocation,
)

__all__ = ["PCA", "TruncatedSVD", "NMF", "FastICA", "KernelPCA",
           "SparsePCA", "DictionaryLearning", "SparseCoder", "MiniBatchNMF",
           "LatentDirichletAllocation"]
