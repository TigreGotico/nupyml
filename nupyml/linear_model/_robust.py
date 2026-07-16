"""Robust regressors: Huber, Quantile, Theil-Sen, RANSAC."""
import itertools

import numpy as np
import scipy.optimize

from ..base import BaseEstimator, RegressorMixin, clone, check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


class HuberRegressor(BaseEstimator, RegressorMixin):
    """Huber loss with a jointly-fitted scale: quadratic near zero, linear in
    the tails, so outliers pull with bounded force."""

    def __init__(self, epsilon=1.35, alpha=1e-4, max_iter=200, tol=1e-6,
                 fit_intercept=True):
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, y_numeric=True)
        n, d = X.shape
        w = np.ones(n) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)

        def loss_grad(theta):
            coef = theta[:d]
            intercept = theta[d] if self.fit_intercept else 0.0
            sigma = theta[-1]
            resid = y - (X @ coef + intercept)
            r = resid / sigma
            small = np.abs(r) <= self.epsilon
            # Huber's concomitant-scale objective (Owen 2006)
            loss_small = w * (sigma + resid ** 2 / sigma)
            loss_large = w * (sigma + 2 * self.epsilon * np.abs(resid)
                              - self.epsilon ** 2 * sigma)
            loss = np.where(small, loss_small, loss_large).sum() \
                + self.alpha * (coef ** 2).sum()
            # gradients
            dresid = np.where(small, -2 * w * resid / sigma,
                              -2 * self.epsilon * w * np.sign(resid))
            gcoef = X.T @ dresid + 2 * self.alpha * coef
            grad = [gcoef]
            if self.fit_intercept:
                grad.append(np.array([dresid.sum()]))
            dsigma = np.where(small, w * (1 - (resid / sigma) ** 2),
                              w * (1 - self.epsilon ** 2)).sum()
            grad.append(np.array([dsigma]))
            return loss, np.concatenate(grad)

        n_params = d + (1 if self.fit_intercept else 0) + 1
        theta0 = np.zeros(n_params)
        theta0[-1] = max(np.std(y), 1.0)
        bounds = [(None, None)] * (n_params - 1) + [(1e-8, None)]
        res = scipy.optimize.minimize(loss_grad, theta0, jac=True,
                                      method="L-BFGS-B", bounds=bounds,
                                      options={"maxiter": self.max_iter,
                                               "gtol": self.tol})
        self.coef_ = res.x[:d]
        self.intercept_ = float(res.x[d]) if self.fit_intercept else 0.0
        self.scale_ = float(res.x[-1])
        resid = y - (X @ self.coef_ + self.intercept_)
        self.outliers_ = np.abs(resid / self.scale_) > self.epsilon
        self.n_iter_ = res.nit
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


class QuantileRegressor(BaseEstimator, RegressorMixin):
    """Pinball-loss regression, solved as a linear program."""

    def __init__(self, quantile=0.5, alpha=1.0, fit_intercept=True):
        self.quantile = quantile
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, y_numeric=True)
        n, d = X.shape
        w = np.ones(n) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        q = self.quantile
        # variables: coef+ (d), coef- (d), [intercept+, intercept-], u (n), v (n)
        n_int = 1 if self.fit_intercept else 0
        n_vars = 2 * d + 2 * n_int + 2 * n
        c = np.concatenate([
            np.full(2 * d, self.alpha * n),          # L1 penalty on coefficients
            np.zeros(2 * n_int),                     # intercept is unpenalized
            q * w, (1 - q) * w,                      # pinball residual costs
        ])
        blocks = [X, -X]
        if self.fit_intercept:
            blocks += [np.ones((n, 1)), -np.ones((n, 1))]
        blocks += [np.eye(n), -np.eye(n)]
        A_eq = np.hstack(blocks)
        res = scipy.optimize.linprog(
            c, A_eq=A_eq, b_eq=y, bounds=[(0, None)] * n_vars,
            method="highs")
        if not res.success:
            raise RuntimeError(f"Quantile LP failed: {res.message}")
        sol = res.x
        self.coef_ = sol[:d] - sol[d:2 * d]
        if self.fit_intercept:
            self.intercept_ = float(sol[2 * d] - sol[2 * d + 1])
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


class TheilSenRegressor(BaseEstimator, RegressorMixin):
    """Spatial median of least-squares fits over random subsets: breaks down
    only when ~29% of the data is contaminated."""

    def __init__(self, n_subsamples=None, max_subpopulation=1000,
                 fit_intercept=True, max_iter=300, tol=1e-3, random_state=None):
        self.n_subsamples = n_subsamples
        self.max_subpopulation = max_subpopulation
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _spatial_median(points, max_iter, tol):
        """Weiszfeld's algorithm for the L1 (geometric) median."""
        x = np.median(points, axis=0)
        for _ in range(max_iter):
            d = np.linalg.norm(points - x, axis=1)
            nonzero = d > 1e-12
            if not nonzero.any():
                return x
            w = 1.0 / d[nonzero]
            x_new = (points[nonzero] * w[:, None]).sum(axis=0) / w.sum()
            if np.linalg.norm(x_new - x) < tol:
                return x_new
            x = x_new
        return x

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        n, d = X.shape
        n_sub = self.n_subsamples or (d + 1 if self.fit_intercept else d)
        n_sub = max(n_sub, d + (1 if self.fit_intercept else 0))
        # enumerate all subsets when cheap, else sample
        from math import comb
        total = comb(n, n_sub) if n_sub <= n else 0
        if 0 < total <= self.max_subpopulation:
            subsets = list(itertools.combinations(range(n), n_sub))
        else:
            subsets = [rng.choice(n, size=n_sub, replace=False)
                       for _ in range(self.max_subpopulation)]
        fits = []
        for idx in subsets:
            idx = np.asarray(idx)
            Xs = X[idx]
            if self.fit_intercept:
                Xs = np.hstack([Xs, np.ones((len(idx), 1))])
            try:
                coef, *_ = np.linalg.lstsq(Xs, y[idx], rcond=None)
            except np.linalg.LinAlgError:
                continue
            if np.all(np.isfinite(coef)):
                fits.append(coef)
        if not fits:
            raise RuntimeError("No valid subset fit could be computed")
        fits = np.asarray(fits)
        median = self._spatial_median(fits, self.max_iter, self.tol)
        self.coef_ = median[:d]
        self.intercept_ = float(median[d]) if self.fit_intercept else 0.0
        self.n_subpopulation_ = len(fits)
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


class RANSACRegressor(BaseEstimator, RegressorMixin):
    """Random sample consensus: fit on minimal subsets, keep the model with
    the largest inlier set."""

    def __init__(self, estimator=None, min_samples=None, residual_threshold=None,
                 max_trials=100, stop_probability=0.99, random_state=None):
        self.estimator = estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.stop_probability = stop_probability
        self.random_state = random_state

    def fit(self, X, y):
        from .import LinearRegression
        X, y = check_X_y(X, y, y_numeric=True)
        rng = check_random_state(self.random_state)
        n, d = X.shape
        base = self.estimator if self.estimator is not None else LinearRegression()
        min_samples = self.min_samples or (d + 1)
        if isinstance(min_samples, float):
            min_samples = max(1, int(min_samples * n))
        # MAD is the standard scale-free default for the inlier threshold
        threshold = self.residual_threshold if self.residual_threshold is not None \
            else np.median(np.abs(y - np.median(y)))
        best_inliers = None
        best_score = -np.inf
        trials = 0
        for trials in range(1, self.max_trials + 1):
            idx = rng.choice(n, size=min_samples, replace=False)
            if len(np.unique(X[idx], axis=0)) < min_samples:
                continue
            model = clone(base).fit(X[idx], y[idx])
            resid = np.abs(y - model.predict(X))
            inliers = resid <= threshold
            n_in = int(inliers.sum())
            if n_in < d + 1:
                continue
            # break ties on inlier count by residual sum
            score = n_in - resid[inliers].sum() / (n_in * max(threshold, 1e-12))
            if score > best_score:
                best_score, best_inliers = score, inliers
                # early exit once the odds of a better sample are negligible
                ratio = n_in / n
                if ratio > 0:
                    with np.errstate(divide="ignore"):
                        needed = (np.log(1 - self.stop_probability)
                                  / np.log(1 - ratio ** min_samples + 1e-12))
                    if trials >= needed:
                        break
        if best_inliers is None:
            raise RuntimeError(
                "RANSAC could not find a valid consensus set; try raising "
                "residual_threshold or max_trials")
        self.inlier_mask_ = best_inliers
        self.estimator_ = clone(base).fit(X[best_inliers], y[best_inliers])
        self.n_trials_ = trials
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(check_array(X))

    @property
    def coef_(self):
        check_is_fitted(self, "estimator_")
        return self.estimator_.coef_

    @property
    def intercept_(self):
        check_is_fitted(self, "estimator_")
        return self.estimator_.intercept_


__all__ = ["HuberRegressor", "QuantileRegressor", "TheilSenRegressor",
           "RANSACRegressor"]
