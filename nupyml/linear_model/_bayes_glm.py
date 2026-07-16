"""Bayesian linear models, GLMs, and built-in regularization-path CV."""
import numpy as np
import scipy.optimize
import scipy.linalg

from ..base import BaseEstimator, RegressorMixin, ClassifierMixin, clone, \
    check_is_fitted
from ..utils import check_X_y, check_array, check_random_state


class BayesianRidge(BaseEstimator, RegressorMixin):
    """Ridge with alpha/lambda learned by evidence maximization."""

    def __init__(self, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6,
                 lambda_1=1e-6, lambda_2=1e-6, fit_intercept=True):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        n, d = X.shape
        if self.fit_intercept:
            X_mean, y_mean = X.mean(axis=0), y.mean()
            Xc, yc = X - X_mean, y - y_mean
        else:
            Xc, yc = X, y
        alpha = 1.0 / (np.var(yc) + 1e-12)     # noise precision
        lam = 1.0                              # weight precision
        XtX = Xc.T @ Xc
        Xty = Xc.T @ yc
        eigvals = np.linalg.eigvalsh(XtX)
        coef_old = None
        for it in range(self.max_iter):
            # posterior over the weights
            A = alpha * XtX + lam * np.eye(d)
            coef = np.linalg.solve(A, alpha * Xty)
            # effective number of well-determined parameters
            gamma = np.sum(alpha * eigvals / (alpha * eigvals + lam))
            rmse = np.sum((yc - Xc @ coef) ** 2)
            lam = ((gamma + 2 * self.lambda_1)
                   / (np.sum(coef ** 2) + 2 * self.lambda_2))
            alpha = ((n - gamma + 2 * self.alpha_1)
                     / (rmse + 2 * self.alpha_2))
            if coef_old is not None and np.max(np.abs(coef - coef_old)) < self.tol:
                break
            coef_old = coef.copy()
        self.alpha_ = float(alpha)
        self.lambda_ = float(lam)
        self.coef_ = coef
        self.sigma_ = np.linalg.inv(alpha * XtX + lam * np.eye(d))
        self.intercept_ = float(y_mean - X_mean @ coef) if self.fit_intercept \
            else 0.0
        self.n_iter_ = it + 1
        self._X_offset = X_mean if self.fit_intercept else np.zeros(d)
        return self

    def predict(self, X, return_std=False):
        check_is_fitted(self, "coef_")
        X = check_array(X)
        mean = X @ self.coef_ + self.intercept_
        if not return_std:
            return mean
        Xc = X - self._X_offset
        var = 1.0 / self.alpha_ + np.einsum("ij,jk,ik->i", Xc, self.sigma_, Xc)
        return mean, np.sqrt(var)


class ARDRegression(BaseEstimator, RegressorMixin):
    """Automatic relevance determination: a per-feature prior precision, so
    irrelevant features are pruned to exactly zero weight."""

    def __init__(self, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6,
                 lambda_1=1e-6, lambda_2=1e-6, threshold_lambda=1e4,
                 fit_intercept=True):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.threshold_lambda = threshold_lambda
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        n, d = X.shape
        if self.fit_intercept:
            X_mean, y_mean = X.mean(axis=0), y.mean()
            Xc, yc = X - X_mean, y - y_mean
        else:
            Xc, yc = X, y
        alpha = 1.0 / (np.var(yc) + 1e-12)
        lam = np.ones(d)
        coef = np.zeros(d)
        keep = np.ones(d, dtype=bool)
        for it in range(self.max_iter):
            keep = lam < self.threshold_lambda
            if not keep.any():
                break
            Xk = Xc[:, keep]
            A = alpha * (Xk.T @ Xk) + np.diag(lam[keep])
            sigma = np.linalg.inv(A)
            coef_k = alpha * sigma @ (Xk.T @ yc)
            gamma = 1.0 - lam[keep] * np.diag(sigma)
            lam_new = np.full(d, np.inf)
            lam_new[keep] = ((gamma + 2 * self.lambda_1)
                             / (coef_k ** 2 + 2 * self.lambda_2))
            rmse = np.sum((yc - Xk @ coef_k) ** 2)
            alpha = ((n - gamma.sum() + 2 * self.alpha_1)
                     / (rmse + 2 * self.alpha_2))
            coef_new = np.zeros(d)
            coef_new[keep] = coef_k
            if np.max(np.abs(coef_new - coef)) < self.tol:
                coef, lam = coef_new, lam_new
                break
            coef, lam = coef_new, lam_new
        self.coef_ = coef
        self.lambda_ = lam
        self.alpha_ = float(alpha)
        self.sigma_ = sigma
        self.intercept_ = float(y_mean - X_mean @ coef) if self.fit_intercept \
            else 0.0
        self.n_iter_ = it + 1
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


class _GLM(BaseEstimator, RegressorMixin):
    """Generalized linear model with a Tweedie deviance and a log/identity link."""

    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=200, tol=1e-8):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    _power = 0.0   # subclasses set the Tweedie power
    _link = "log"

    def _resolve_link(self):
        """'auto' means identity for the normal case and log otherwise, which
        is what makes power=0 plain least squares."""
        link = self._link
        if link == "auto":
            return "identity" if self._power == 0 else "log"
        return link

    def _deviance_grad(self, theta, X, y, w, d):
        coef = theta[:d]
        b = theta[d] if self.fit_intercept else 0.0
        eta = X @ coef + b
        link = self._resolve_link()
        mu = eta if link == "identity" else np.exp(eta)
        p = self._power
        if p == 0:            # normal
            dev = w * (y - mu) ** 2
            dmu = -2 * w * (y - mu)
        elif p == 1:          # poisson
            with np.errstate(divide="ignore", invalid="ignore"):
                term = np.where(y > 0, y * np.log(y / mu), 0.0)
            dev = 2 * w * (term - y + mu)
            dmu = 2 * w * (1 - y / mu)
        elif p == 2:          # gamma
            dev = 2 * w * (np.log(mu / y) + y / mu - 1)
            dmu = 2 * w * (1 / mu - y / mu ** 2)
        else:                 # general tweedie
            dev = 2 * w * (np.power(np.maximum(y, 0), 2 - p) / ((1 - p) * (2 - p))
                           - y * np.power(mu, 1 - p) / (1 - p)
                           + np.power(mu, 2 - p) / (2 - p))
            dmu = 2 * w * (-y * np.power(mu, -p) + np.power(mu, 1 - p))
        # chain rule through the link: dmu/deta is mu for log, 1 for identity
        deta = dmu if link == "identity" else dmu * mu
        # the data term is averaged over the total sample weight and the
        # penalty is 0.5*alpha*||w||^2, so alpha means what it does in sklearn
        sw_sum = w.sum()
        loss = dev.sum() / (2 * sw_sum) + 0.5 * self.alpha * (coef ** 2).sum()
        gcoef = X.T @ deta / (2 * sw_sum) + self.alpha * coef
        grad = [gcoef]
        if self.fit_intercept:
            grad.append(np.array([deta.sum() / (2 * sw_sum)]))
        return loss, np.concatenate(grad)

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, y_numeric=True)
        p = self._power
        # the Tweedie family's support depends on the power: normal (p=0)
        # spans the reals, Poisson/compound-Poisson need y >= 0, and gamma
        # and beyond need y > 0
        if 1 <= p < 2 and (y < 0).any():
            raise ValueError(f"{type(self).__name__} requires y >= 0 for "
                             f"power={p}")
        if p >= 2 and (y <= 0).any():
            raise ValueError(f"{type(self).__name__} requires strictly "
                             f"positive y for power={p}")
        n, d = X.shape
        w = np.ones(n) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float64)
        n_params = d + (1 if self.fit_intercept else 0)
        theta0 = np.zeros(n_params)
        if self.fit_intercept:
            theta0[d] = (y.mean() if self._resolve_link() == "identity"
                         else np.log(max(y.mean(), 1e-6)))
        res = scipy.optimize.minimize(
            self._deviance_grad, theta0, args=(X, y, w, d), jac=True,
            method="L-BFGS-B", options={"maxiter": self.max_iter,
                                        "gtol": self.tol})
        self.coef_ = res.x[:d]
        self.intercept_ = float(res.x[d]) if self.fit_intercept else 0.0
        self.n_iter_ = res.nit
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        eta = check_array(X) @ self.coef_ + self.intercept_
        return eta if self._resolve_link() == "identity" else np.exp(eta)

    def score(self, X, y, sample_weight=None):
        """D^2, the deviance analogue of R^2."""
        y = np.asarray(y, dtype=np.float64)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
        d = X.shape[1]
        pred = self.predict(X)
        dev = self._deviance_of(y, pred, w)
        null = self._deviance_of(y, np.full(len(y), np.average(y, weights=w)), w)
        return float(1 - dev / null) if null > 0 else 0.0

    def _deviance_of(self, y, mu, w):
        p = self._power
        if p == 0:
            return float((w * (y - mu) ** 2).sum())
        if p == 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                term = np.where(y > 0, y * np.log(y / mu), 0.0)
            return float((2 * w * (term - y + mu)).sum())
        if p == 2:
            return float((2 * w * (np.log(mu / y) + y / mu - 1)).sum())
        return float((2 * w * (np.power(y, 2 - p) / ((1 - p) * (2 - p))
                               - y * np.power(mu, 1 - p) / (1 - p)
                               + np.power(mu, 2 - p) / (2 - p))).sum())


class PoissonRegressor(_GLM):
    _power = 1.0
    _link = "log"
    _estimator_tags = {"requires_positive_y": True}


class GammaRegressor(_GLM):
    _power = 2.0
    _link = "log"
    _estimator_tags = {"requires_positive_y": True}


class TweedieRegressor(_GLM):
    @property
    def _estimator_tags(self):
        return {"requires_positive_y": self.power >= 1}

    def __init__(self, power=0.0, alpha=1.0, link="auto", fit_intercept=True,
                 max_iter=200, tol=1e-8):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept,
                         max_iter=max_iter, tol=tol)
        self.power = power
        self.link = link

    @property
    def _power(self):
        return self.power

    @property
    def _link(self):
        return self.link


# ---------------------------------------------------------------------------
# built-in cross-validated regularization paths
# ---------------------------------------------------------------------------

class _PathCV(BaseEstimator, RegressorMixin):
    def _alphas(self, X, y):
        if self.alphas is not None:
            return np.asarray(self.alphas, dtype=np.float64)
        # geometric grid down from the smallest alpha that zeroes every coef
        alpha_max = np.abs(X.T @ (y - y.mean())).max() / len(X)
        alpha_max = max(alpha_max, 1e-3)
        return np.logspace(np.log10(alpha_max), np.log10(alpha_max * 1e-3),
                           self.n_alphas)

    def fit(self, X, y):
        from ..model_selection import _check_cv
        X, y = check_X_y(X, y, y_numeric=True)
        alphas = self._alphas(X, y)
        cv = _check_cv(self.cv, y, classifier=False)
        scores = np.zeros((len(alphas), cv.get_n_splits(X, y)))
        for j, (train, test) in enumerate(cv.split(X, y)):
            for i, a in enumerate(alphas):
                est = self._make(a).fit(X[train], y[train])
                scores[i, j] = -np.mean((est.predict(X[test]) - y[test]) ** 2)
        self.alphas_ = alphas
        self.mse_path_ = -scores
        self.alpha_ = float(alphas[np.argmax(scores.mean(axis=1))])
        final = self._make(self.alpha_).fit(X, y)
        self.coef_ = final.coef_
        self.intercept_ = final.intercept_
        self._final = final
        return self

    def predict(self, X):
        check_is_fitted(self, "alpha_")
        return self._final.predict(X)


class RidgeCV(_PathCV):
    def __init__(self, alphas=(0.1, 1.0, 10.0), fit_intercept=True, cv=5):
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.n_alphas = 100

    def _make(self, alpha):
        from . import Ridge
        return Ridge(alpha=alpha, fit_intercept=self.fit_intercept)


class LassoCV(_PathCV):
    def __init__(self, alphas=None, n_alphas=20, fit_intercept=True, cv=5,
                 max_iter=1000, tol=1e-4):
        self.alphas = alphas
        self.n_alphas = n_alphas
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.max_iter = max_iter
        self.tol = tol

    def _make(self, alpha):
        from . import Lasso
        return Lasso(alpha=alpha, fit_intercept=self.fit_intercept,
                     max_iter=self.max_iter, tol=self.tol)


class ElasticNetCV(_PathCV):
    def __init__(self, l1_ratio=0.5, alphas=None, n_alphas=20,
                 fit_intercept=True, cv=5, max_iter=1000, tol=1e-4):
        self.l1_ratio = l1_ratio
        self.alphas = alphas
        self.n_alphas = n_alphas
        self.fit_intercept = fit_intercept
        self.cv = cv
        self.max_iter = max_iter
        self.tol = tol

    def _make(self, alpha):
        from . import ElasticNet
        return ElasticNet(alpha=alpha, l1_ratio=self.l1_ratio,
                          fit_intercept=self.fit_intercept,
                          max_iter=self.max_iter, tol=self.tol)


class LogisticRegressionCV(BaseEstimator, ClassifierMixin):
    def __init__(self, Cs=10, cv=5, fit_intercept=True, max_iter=200,
                 scoring=None):
        self.Cs = Cs
        self.cv = cv
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.scoring = scoring

    def fit(self, X, y):
        from . import LogisticRegression
        from ..model_selection import _check_cv, cross_val_score
        X, y = check_X_y(X, y)
        Cs = (np.logspace(-4, 4, self.Cs) if isinstance(self.Cs, int)
              else np.asarray(self.Cs, dtype=np.float64))
        cv = _check_cv(self.cv, y, classifier=True)
        scores = np.zeros((len(Cs), cv.get_n_splits(X, y)))
        for j, (train, test) in enumerate(cv.split(X, y)):
            for i, C in enumerate(Cs):
                est = LogisticRegression(C=C, fit_intercept=self.fit_intercept,
                                         max_iter=self.max_iter).fit(X[train],
                                                                     y[train])
                scores[i, j] = est.score(X[test], y[test])
        self.Cs_ = Cs
        self.scores_ = scores
        self.C_ = float(Cs[np.argmax(scores.mean(axis=1))])
        self._final = LogisticRegression(
            C=self.C_, fit_intercept=self.fit_intercept,
            max_iter=self.max_iter).fit(X, y)
        self.classes_ = self._final.classes_
        self.coef_ = self._final.coef_
        self.intercept_ = self._final.intercept_
        return self

    def predict(self, X):
        check_is_fitted(self, "C_")
        return self._final.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "C_")
        return self._final.predict_proba(X)


class OrthogonalMatchingPursuit(BaseEstimator, RegressorMixin):
    """Greedy forward selection on the residual correlation."""

    def __init__(self, n_nonzero_coefs=None, tol=None, fit_intercept=True):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        n, d = X.shape
        if self.fit_intercept:
            X_mean, y_mean = X.mean(axis=0), y.mean()
            Xc, yc = X - X_mean, y - y_mean
        else:
            Xc, yc = X, y
        k = self.n_nonzero_coefs or max(1, int(0.1 * d))
        norms = np.linalg.norm(Xc, axis=0)
        norms[norms == 0] = 1.0
        Xn = Xc / norms
        residual = yc.copy()
        selected = []
        coef = np.zeros(d)
        for _ in range(min(k, d)):
            corr = np.abs(Xn.T @ residual)
            corr[selected] = -np.inf
            j = int(np.argmax(corr))
            selected.append(j)
            sub, *_ = np.linalg.lstsq(Xc[:, selected], yc, rcond=None)
            residual = yc - Xc[:, selected] @ sub
            if self.tol is not None and np.sum(residual ** 2) < self.tol:
                break
        coef[selected] = sub
        self.coef_ = coef
        self.n_nonzero_coefs_ = len(selected)
        self.intercept_ = float(y_mean - X_mean @ coef) if self.fit_intercept \
            else 0.0
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


__all__ = ["BayesianRidge", "ARDRegression", "PoissonRegressor",
           "GammaRegressor", "TweedieRegressor", "RidgeCV", "LassoCV",
           "ElasticNetCV", "LogisticRegressionCV", "OrthogonalMatchingPursuit"]
