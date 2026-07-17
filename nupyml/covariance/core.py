"""Covariance estimators: empirical, shrunk, sparse, and robust."""
import numpy as np

from ..base import BaseEstimator, check_is_fitted
from ..utils import check_array


class EmpiricalCovariance(BaseEstimator):
    """The plain sample covariance -- the baseline the others improve on.

    ``(1/n) * X_centered^T X_centered``. Unbiased, and the maximum-likelihood
    estimate for a Gaussian -- but high-variance, singular when features exceed
    samples, and with an inverse that amplifies its noisiest directions. Here as
    the reference point; prefer ``LedoitWolf`` unless n vastly exceeds the number
    of features.
    """

    def fit(self, X, y=None):
        X = check_array(X)
        self.location_ = X.mean(axis=0)
        Xc = X - self.location_
        self.covariance_ = (Xc.T @ Xc) / len(X)
        return self

    def mahalanobis(self, X):
        """Squared Mahalanobis distance -- covariance-aware distance to the centre.

        This is what covariance is usually FOR: a distance that accounts for the
        spread and correlation of the data, so an outlier is measured in units of
        the data's own variability rather than raw coordinates.
        """
        check_is_fitted(self, "covariance_")
        Xc = check_array(X) - self.location_
        prec = np.linalg.pinv(self.covariance_)     # pinv: survives singularity
        return np.sum((Xc @ prec) * Xc, axis=1)


class ShrunkCovariance(EmpiricalCovariance):
    """Blend the sample covariance with a scaled identity, at a FIXED intensity.

    ``(1 - alpha) * sample + alpha * (trace/p) * I``

    Shrinking toward the identity pulls the extreme eigenvalues in and pushes the
    tiny ones up, curing both the singularity and the inverse's noise
    amplification. ``alpha`` is the dial: 0 is the raw sample covariance, 1 is a
    pure scaled identity. ``LedoitWolf`` is this with ``alpha`` chosen optimally
    instead of guessed.
    """

    def __init__(self, shrinkage=0.1):
        self.shrinkage = shrinkage

    def fit(self, X, y=None):
        X = check_array(X)
        self.location_ = X.mean(axis=0)
        Xc = X - self.location_
        sample = (Xc.T @ Xc) / len(X)
        p = X.shape[1]
        mu = np.trace(sample) / p                    # the identity target's scale
        self.covariance_ = ((1 - self.shrinkage) * sample
                            + self.shrinkage * mu * np.eye(p))
        return self


class LedoitWolf(EmpiricalCovariance):
    """Shrinkage toward the identity, at the PROVABLY optimal intensity.

    THE INSIGHT
    -----------
    Shrinkage always helps, but how much? Ledoit and Wolf derived the ``alpha``
    that minimises the expected squared error between the estimate and the true
    covariance -- a closed-form quantity computed directly from the data, no
    cross-validation. It is essentially the ratio of the sample covariance's
    VARIANCE (how noisy it is) to how far it sits from the identity target: noisy
    estimates get shrunk hard, reliable ones barely at all.

    This is Stein's paradox made practical: the naive unbiased estimator is
    inadmissible, and a biased one that borrows strength toward a simple target
    beats it, provably. The optimal ``shrinkage_`` is reported so you can see how
    much the data demanded.

    Ledoit & Wolf (2004).
    """

    def fit(self, X, y=None):
        X = check_array(X)
        n, p = X.shape
        self.location_ = X.mean(axis=0)
        Xc = X - self.location_
        sample = (Xc.T @ Xc) / n
        mu = np.trace(sample) / p
        target = mu * np.eye(p)

        # the optimal intensity: (variance of the sample cov) / (distance to target)
        # numerator: how much the per-sample covariances scatter around the mean
        d2 = np.sum((sample - target) ** 2)          # ||sample - target||^2
        b2 = 0.0
        for i in range(n):
            xi = Xc[i][:, None]
            b2 += np.sum((xi @ xi.T - sample) ** 2)
        b2 = b2 / n ** 2
        b2 = min(b2, d2)                             # b2 cannot exceed d2
        self.shrinkage_ = float(b2 / d2) if d2 > 0 else 0.0
        self.covariance_ = ((1 - self.shrinkage_) * sample
                            + self.shrinkage_ * target)
        return self


class OAS(EmpiricalCovariance):
    """Oracle Approximating Shrinkage: a shrinkage formula tuned for Gaussians.

    Same shape as Ledoit-Wolf -- shrink toward a scaled identity -- but with an
    intensity derived under the assumption that the data IS Gaussian. When that
    assumption roughly holds (as it often does), OAS chooses a slightly better
    intensity than the distribution-free Ledoit-Wolf and gives a lower-error
    estimate, especially in the small-sample regime it was designed for. The
    trade is exactly that assumption: Ledoit-Wolf is safer when the data is not
    Gaussian, OAS is sharper when it is.

    Chen, Wiesel, Eldar & Hero (2010).
    """

    def fit(self, X, y=None):
        X = check_array(X)
        n, p = X.shape
        self.location_ = X.mean(axis=0)
        Xc = X - self.location_
        sample = (Xc.T @ Xc) / n
        mu = np.trace(sample) / p

        tr = np.trace(sample)
        tr2 = np.trace(sample @ sample)
        # the OAS closed form for the shrinkage, assuming Gaussian data
        num = (1 - 2 / p) * tr2 + tr ** 2
        den = (n + 1 - 2 / p) * (tr2 - tr ** 2 / p)
        self.shrinkage_ = 1.0 if den == 0 else float(min(num / den, 1.0))
        self.covariance_ = ((1 - self.shrinkage_) * sample
                            + self.shrinkage_ * mu * np.eye(p))
        return self


class GraphicalLasso(BaseEstimator):
    """A SPARSE precision matrix: learn the graph of direct relationships.

    WHY THE INVERSE, AND WHY SPARSE
    -------------------------------
    The PRECISION matrix (inverse covariance) is more informative than the
    covariance itself: a zero in it means two variables are CONDITIONALLY
    independent -- unrelated once every other variable is accounted for. That
    distinguishes a direct link from one mediated by a third variable, which plain
    correlation cannot.

    An L1 penalty on the precision matrix drives most of its off-diagonals to
    exactly zero, so the fitted precision is a sparse GRAPH of the variables'
    direct dependencies -- the same lasso sparsity idea, applied to a matrix. It
    is how gene-regulatory and functional-connectivity networks are estimated.

    This is a compact coordinate-descent version of the graphical lasso; the full
    algorithm alternates lasso regressions across the variables.

    Friedman, Hastie & Tibshirani (2008).
    """

    def __init__(self, alpha=0.1, max_iter=100, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        X = check_array(X)
        self.location_ = X.mean(axis=0)
        Xc = X - self.location_
        S = (Xc.T @ Xc) / len(X)
        p = X.shape[1]

        # start from a ridged sample covariance so it is invertible
        W = S + self.alpha * np.eye(p)
        for _ in range(self.max_iter):
            W_old = W.copy()
            for j in range(p):
                # block coordinate descent: solve a lasso for column j against the
                # rest, which is where the sparsity enters
                idx = [k for k in range(p) if k != j]
                W11 = W[np.ix_(idx, idx)]
                s12 = S[idx, j]
                beta = self._lasso(W11, s12, self.alpha)
                w12 = W11 @ beta
                W[idx, j] = w12
                W[j, idx] = w12
            if np.max(np.abs(W - W_old)) < self.tol:
                break

        self.covariance_ = W
        self.precision_ = np.linalg.pinv(W)
        return self

    def _lasso(self, A, b, alpha, n_iter=100):
        """Cyclic coordinate descent with soft-thresholding -- the lasso core."""
        beta = np.zeros(A.shape[1])
        diag = np.diag(A)
        for _ in range(n_iter):
            for k in range(len(beta)):
                if diag[k] == 0:
                    continue
                r = b[k] - A[k] @ beta + A[k, k] * beta[k]
                beta[k] = np.sign(r) * max(abs(r) - alpha, 0) / diag[k]
        return beta


class MinCovDet(BaseEstimator):
    """Minimum Covariance Determinant: fit the covariance to the CLEAN core.

    THE ROBUSTNESS IDEA
    -------------------
    The sample covariance has a breakdown point of zero -- a single far outlier
    inflates it arbitrarily, and every covariance-based method downstream inherits
    the damage. MinCovDet finds the subset of ``h`` points (roughly half) whose
    covariance has the SMALLEST DETERMINANT -- the most tightly concentrated
    subset -- and estimates from those, ignoring the rest.

    The determinant measures the volume the covariance ellipsoid encloses;
    minimising it finds the densest cluster of points, which is the outlier-free
    core. Its breakdown point approaches 50%, the best possible. This is the
    covariance analogue of Least Trimmed Squares in ``nonparametric`` -- same
    "fit the tightest half" logic, and the same random-subset-plus-refinement
    (FAST-MCD) to make the combinatorial search tractable.

    Rousseeuw (1984); Rousseeuw & Van Driessen (1999).
    """

    def __init__(self, support_fraction=None, n_trials=30, random_state=None):
        self.support_fraction = support_fraction
        self.n_trials = n_trials
        self.random_state = random_state

    def fit(self, X, y=None):
        from ..utils import check_random_state
        X = check_array(X)
        rng = check_random_state(self.random_state)
        n, p = X.shape
        h = int(self.support_fraction * n) if self.support_fraction \
            else (n + p + 1) // 2               # the high-breakdown default

        best_det, best_support = np.inf, None
        for _ in range(self.n_trials):
            support = rng.choice(n, h, replace=False)
            for _ in range(20):
                # C-step: fit on the current subset, then keep the h points
                # closest to it. Provably lowers the determinant, like LTS
                Xs = X[support]
                loc = Xs.mean(axis=0)
                cov = np.cov(Xs.T) + 1e-6 * np.eye(p)
                d = np.sum(((X - loc) @ np.linalg.pinv(cov)) * (X - loc), axis=1)
                new_support = np.argsort(d)[:h]
                if set(new_support) == set(support):
                    break
                support = new_support
            det = np.linalg.det(np.cov(X[support].T) + 1e-9 * np.eye(p))
            if det < best_det:
                best_det, best_support = det, support

        self.support_ = np.zeros(n, dtype=bool)
        self.support_[best_support] = True
        self.location_ = X[best_support].mean(axis=0)
        self.covariance_ = np.cov(X[best_support].T)
        return self


__all__ = ["EmpiricalCovariance", "LedoitWolf", "OAS", "GraphicalLasso",
           "MinCovDet", "ShrunkCovariance"]
