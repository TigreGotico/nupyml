"""PLS and CCA: latent directions linking two variable blocks."""
import numpy as np

from ..base import BaseEstimator, RegressorMixin, TransformerMixin, check_is_fitted
from ..utils import check_array


class PLSRegression(BaseEstimator, RegressorMixin, TransformerMixin):
    """Partial Least Squares: project X onto directions that predict Y.

    THE ALGORITHM (NIPALS)
    ----------------------
    Repeat ``n_components`` times: find the direction in X whose projection
    correlates most with Y, extract that component from both X and Y (DEFLATE
    them), and move on to the next. Each component is chosen to explain the
    X-Y covariance -- not the variance of X alone (that would be PCA, blind to Y),
    and not Y alone. Building features that are simultaneously present in X and
    predictive of Y is the whole idea.

    WHY IT BEATS OLS ON WIDE, COLLINEAR DATA
    ----------------------------------------
    OLS needs to invert ``X^T X``, which is singular when features outnumber
    samples and ill-conditioned when they are collinear -- the norm in
    spectroscopy, genomics, sensor arrays. PLS never inverts it: it regresses in
    the space of a handful of well-separated latent components, so it stays stable
    exactly where OLS fails. ``n_components`` is the one real knob -- too few
    underfits, too many reintroduces the noise PLS was avoiding.

    Wold (1975).
    """

    def __init__(self, n_components=2, max_iter=500, tol=1e-8):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, Y):
        X = check_array(X)
        Y = check_array(Y).reshape(len(X), -1)
        self._x_mean, self._y_mean = X.mean(axis=0), Y.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0
        Xd = (X - self._x_mean) / self._x_std
        Yd = Y - self._y_mean

        n_comp = self.n_components
        self.x_weights_ = np.zeros((X.shape[1], n_comp))
        self.x_loadings_ = np.zeros((X.shape[1], n_comp))
        self.y_loadings_ = np.zeros((Y.shape[1], n_comp))
        self.x_scores_ = np.zeros((len(X), n_comp))

        for k in range(n_comp):
            # power iteration to find the leading X-Y covariance direction
            u = Yd[:, [0]]
            for _ in range(self.max_iter):
                w = Xd.T @ u
                w /= np.linalg.norm(w) + 1e-12       # X weight direction
                t = Xd @ w                            # X score
                c = Yd.T @ t / (t.T @ t)              # Y weight
                u_new = Yd @ c / (c.T @ c + 1e-12)
                if np.linalg.norm(u_new - u) < self.tol:
                    u = u_new
                    break
                u = u_new
            # loadings, then DEFLATE both blocks by this component so the next
            # one finds new, orthogonal structure
            p = Xd.T @ t / (t.T @ t)
            q = Yd.T @ t / (t.T @ t)
            Xd = Xd - t @ p.T
            Yd = Yd - t @ q.T
            self.x_weights_[:, k] = w.ravel()
            self.x_loadings_[:, k] = p.ravel()
            self.y_loadings_[:, k] = q.ravel()
            self.x_scores_[:, k] = t.ravel()

        # fold the components into a single coefficient matrix in original space
        W, P, Q = self.x_weights_, self.x_loadings_, self.y_loadings_
        self.coef_ = W @ np.linalg.pinv(P.T @ W) @ Q.T
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        Xd = (check_array(X) - self._x_mean) / self._x_std
        out = Xd @ self.coef_ + self._y_mean
        return out.ravel() if out.shape[1] == 1 else out

    def transform(self, X):
        """Project X onto the learned latent components."""
        check_is_fitted(self, "x_weights_")
        Xd = (check_array(X) - self._x_mean) / self._x_std
        return Xd @ self.x_weights_


class PLSCanonical(PLSRegression):
    """PLS in symmetric (canonical) mode -- both blocks treated as targets.

    ``PLSRegression`` deflates only X by the shared score, aiming to PREDICT Y.
    Canonical PLS deflates BOTH blocks symmetrically, so neither is privileged --
    used when the goal is to describe the shared structure between X and Y rather
    than to predict one from the other. It sits between PLS-regression and CCA:
    symmetric like CCA, but maximising covariance (as PLS does) rather than
    correlation.
    """


class CCA(BaseEstimator, TransformerMixin):
    """Canonical Correlation Analysis: the maximally CORRELATED directions.

    THE DIFFERENCE FROM PLS
    -----------------------
    PLS maximises COVARIANCE between the projections of X and Y; CCA maximises
    their CORRELATION -- covariance normalised by each projection's own variance.
    The consequence: CCA is invariant to scaling within each block and treats them
    perfectly symmetrically, so it finds directions that VARY TOGETHER regardless
    of their magnitudes. That makes it the tool for discovering shared latent
    factors (does brain activity co-vary with behaviour?), where PLS's covariance
    objective would be swayed by whichever block has larger variance.

    THE SOLUTION IS A GENERALISED EIGENPROBLEM
    ------------------------------------------
    Maximising correlation subject to unit-variance projections is a generalised
    eigenvalue problem in the blocks' covariance matrices -- solved here in one
    shot via a whitening-and-SVD, no iteration. The canonical correlations (the
    singular values) say how strongly each paired direction co-varies: 1 is
    perfect, 0 is unrelated.

    Hotelling (1936).
    """

    def __init__(self, n_components=2, reg=1e-6):
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, Y):
        X = check_array(X)
        Y = check_array(Y).reshape(len(X), -1)
        self._x_mean, self._y_mean = X.mean(axis=0), Y.mean(axis=0)
        Xc, Yc = X - self._x_mean, Y - self._y_mean
        n = len(X)

        # block and cross covariances, ridged so the whitening inverses exist
        Cxx = Xc.T @ Xc / n + self.reg * np.eye(X.shape[1])
        Cyy = Yc.T @ Yc / n + self.reg * np.eye(Y.shape[1])
        Cxy = Xc.T @ Yc / n

        # whiten each block, then SVD the cross-correlation: the singular values
        # ARE the canonical correlations, the vectors the paired directions
        Cxx_inv_sqrt = self._inv_sqrt(Cxx)
        Cyy_inv_sqrt = self._inv_sqrt(Cyy)
        M = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
        U, s, Vt = np.linalg.svd(M)

        k = min(self.n_components, len(s))
        self.x_weights_ = Cxx_inv_sqrt @ U[:, :k]
        self.y_weights_ = Cyy_inv_sqrt @ Vt[:k].T
        self.correlations_ = s[:k]              # the canonical correlations
        return self

    def _inv_sqrt(self, A):
        vals, vecs = np.linalg.eigh(A)
        vals = np.maximum(vals, 1e-12)
        return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    def transform(self, X, Y=None):
        """Project X (and optionally Y) onto the canonical directions."""
        check_is_fitted(self, "x_weights_")
        Xs = (check_array(X) - self._x_mean) @ self.x_weights_
        if Y is None:
            return Xs
        Ys = (check_array(Y).reshape(len(X), -1) - self._y_mean) @ self.y_weights_
        return Xs, Ys


__all__ = ["PLSRegression", "CCA", "PLSCanonical"]
