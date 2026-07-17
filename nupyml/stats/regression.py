"""Regression WITH inference: standard errors, tests, and intervals.

Every model here fits like the ones in ``linear_model`` but additionally reports
the sampling uncertainty of its estimates. The extra machinery is small -- one
covariance matrix -- and everything a ``.summary()`` prints is derived from it.
"""
import numpy as np
from scipy import stats

from ..base import BaseEstimator, RegressorMixin, check_is_fitted
from ..utils import check_X_y, check_array


def _add_intercept(X):
    return np.column_stack([np.ones(len(X)), X])


class _LinearInferenceMixin:
    """Shared reporting once ``coef_``, its covariance, and residuals exist."""

    def _finalize_linear(self, X, y, resid, cov_beta, n_params):
        n = len(y)
        self.df_resid_ = n - n_params
        self.bse_ = np.sqrt(np.diag(cov_beta))          # standard errors
        # t-statistic = estimate / its standard error; large |t| => far from zero
        self.tvalues_ = self.params_ / self.bse_
        self.pvalues_ = 2 * stats.t.sf(np.abs(self.tvalues_), self.df_resid_)
        self.cov_params_ = cov_beta

        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        self.rsquared_ = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        # adjusted R^2 penalises extra predictors, so it cannot be gamed by
        # adding noise features the way plain R^2 can
        self.rsquared_adj_ = 1 - (1 - self.rsquared_) * (n - 1) / self.df_resid_
        self.scale_ = ss_res / self.df_resid_           # residual variance estimate
        # log-likelihood of the Gaussian model, for AIC/BIC
        self.llf_ = -0.5 * n * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
        self.aic_ = 2 * n_params - 2 * self.llf_
        self.bic_ = n_params * np.log(n) - 2 * self.llf_
        # overall F-test: are the slopes JOINTLY zero? (excludes the intercept)
        k = n_params - (1 if self.fit_intercept else 0)
        if k > 0 and ss_tot > 0 and self.df_resid_ > 0 and self.rsquared_ < 1:
            self.fvalue_ = (self.rsquared_ / k) / \
                ((1 - self.rsquared_) / self.df_resid_)
            self.f_pvalue_ = stats.f.sf(self.fvalue_, k, self.df_resid_)
        elif self.rsquared_ >= 1:
            # a perfect fit: the model explains everything, F is infinite and the
            # p-value is zero -- report that rather than dividing by zero
            self.fvalue_ = np.inf
            self.f_pvalue_ = 0.0

    def conf_int(self, alpha=0.05):
        """Confidence interval per coefficient.

        The interval that would contain the true coefficient ``(1-alpha)`` of the
        time under repeated sampling. Built as ``estimate +- t_crit * SE`` -- a
        wide interval means the data barely pins the coefficient down, and one
        that straddles zero says the effect is not distinguishable from none.
        """
        check_is_fitted(self, "params_")
        t_crit = stats.t.ppf(1 - alpha / 2, self.df_resid_)
        lo = self.params_ - t_crit * self.bse_
        hi = self.params_ + t_crit * self.bse_
        return np.column_stack([lo, hi])

    def summary(self):
        """A statsmodels-style table: estimate, SE, t, p, and CI per coefficient."""
        check_is_fitted(self, "params_")
        ci = self.conf_int()
        names = self._param_names()
        header = (f"{type(self).__name__}   n={self.nobs_}   "
                  f"R2={getattr(self, 'rsquared_', float('nan')):.4f}   "
                  f"adjR2={getattr(self, 'rsquared_adj_', float('nan')):.4f}   "
                  f"AIC={self.aic_:.1f}   BIC={self.bic_:.1f}")
        lines = [header, "-" * len(header),
                 f"{'term':<12}{'coef':>10}{'std err':>10}{'t':>8}"
                 f"{'P>|t|':>9}{'[0.025':>10}{'0.975]':>10}"]
        for i, nm in enumerate(names):
            lines.append(f"{nm:<12}{self.params_[i]:>10.4f}{self.bse_[i]:>10.4f}"
                         f"{self.tvalues_[i]:>8.2f}{self.pvalues_[i]:>9.3f}"
                         f"{ci[i, 0]:>10.4f}{ci[i, 1]:>10.4f}")
        return "\n".join(lines)

    def _param_names(self):
        p = len(self.params_) - (1 if self.fit_intercept else 0)
        base = [f"x{i}" for i in range(p)]
        return (["const"] + base) if self.fit_intercept else base


class OLS(BaseEstimator, RegressorMixin, _LinearInferenceMixin):
    """Ordinary least squares with full inference.

    Fits ``beta = (X'X)^-1 X'y`` -- the same estimate as ``linear_model.
    LinearRegression`` -- but also reports where each coefficient sits in its
    sampling distribution. The coefficient covariance is
    ``sigma^2 (X'X)^-1``, where ``sigma^2`` is estimated from the residuals with
    the ``n - p`` correction (dividing by the residual degrees of freedom, not
    ``n``, so the variance estimate is unbiased).

    ROBUST STANDARD ERRORS
    ----------------------
    The classical covariance assumes the errors are homoskedastic -- equal
    variance everywhere. When they are not (common in economics and any data
    where spread grows with the mean), the point estimates stay unbiased but the
    classical SEs are wrong, so the p-values lie. ``cov_type="HC0".."HC3"`` uses
    the heteroskedasticity-consistent "sandwich" estimator instead, which is valid
    without the equal-variance assumption. HC3 is the small-sample default. This
    is the single most common robustness fix in applied regression.
    """

    def __init__(self, fit_intercept=True, cov_type="nonrobust"):
        self.fit_intercept = fit_intercept
        self.cov_type = cov_type

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.nobs_ = len(y)
        Xd = _add_intercept(X) if self.fit_intercept else X
        # solve the normal equations via pinv (stable if X is rank-deficient)
        XtX_inv = np.linalg.pinv(Xd.T @ Xd)
        self.params_ = XtX_inv @ Xd.T @ y
        resid = y - Xd @ self.params_
        self.resid_ = resid
        n_params = Xd.shape[1]

        if self.cov_type == "nonrobust":
            scale = (resid @ resid) / (len(y) - n_params)
            cov_beta = scale * XtX_inv
        else:
            cov_beta = self._sandwich(Xd, resid, XtX_inv, n_params)

        self.coef_ = self.params_[1:] if self.fit_intercept else self.params_
        self.intercept_ = self.params_[0] if self.fit_intercept else 0.0
        self._Xd = Xd
        self._finalize_linear(Xd, y, resid, cov_beta, n_params)
        return self

    def _sandwich(self, X, resid, XtX_inv, n_params):
        """The HC "sandwich" covariance ``(X'X)^-1 X' diag(w) X (X'X)^-1``.

        The meat ``diag(w)`` weights each observation by its own squared residual
        (with a finite-sample adjustment per HC variant), so a point with a large
        residual widens the SE of the coefficients it influences -- exactly the
        correction homoskedastic SEs miss.
        """
        n = len(resid)
        r2 = resid ** 2
        if self.cov_type == "HC0":
            w = r2
        elif self.cov_type == "HC1":
            w = r2 * n / (n - n_params)
        else:                                   # HC2 / HC3 use the leverage h_ii
            h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
            w = r2 / (1 - h) if self.cov_type == "HC2" else r2 / (1 - h) ** 2
        meat = X.T @ (w[:, None] * X)
        return XtX_inv @ meat @ XtX_inv

    def predict(self, X):
        check_is_fitted(self, "params_")
        Xd = _add_intercept(check_array(X)) if self.fit_intercept else check_array(X)
        return Xd @ self.params_


class WLS(OLS):
    """Weighted least squares: give each observation its own reliability.

    When observations differ in precision -- some measured more noisily than
    others -- OLS wastes information by treating them equally. WLS weights each by
    ``weights`` (ideally the inverse of its error variance), so precise points
    pull the fit more. It is the right estimator when the error variance is known
    up to a per-point factor, and it is OLS on the whitened data
    ``sqrt(w) * (X, y)``, which is exactly how it is implemented here.
    """

    def __init__(self, fit_intercept=True, cov_type="nonrobust"):
        super().__init__(fit_intercept=fit_intercept, cov_type=cov_type)

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, y_numeric=True)
        w = np.ones(len(y)) if sample_weight is None \
            else np.asarray(sample_weight, dtype=float)
        sw = np.sqrt(w)
        # whiten EVERYTHING by sqrt(w), the intercept column included -- scaling
        # only the features (and leaving the ones-column unscaled) mis-fits the
        # intercept and was the bug this guards against. Build the full design
        # here and fit without a second intercept.
        Xd = _add_intercept(X) if self.fit_intercept else X
        Xw = Xd * sw[:, None]
        yw = y * sw
        # temporarily fit the whitened design with no extra intercept column
        want_intercept = self.fit_intercept
        self.fit_intercept = False
        try:
            super().fit(Xw, yw)
        finally:
            self.fit_intercept = want_intercept
        # params_ are in the whitened basis but correspond 1:1 to the original
        # design columns; re-expose coef_/intercept_ against the real layout
        self.coef_ = self.params_[1:] if want_intercept else self.params_
        self.intercept_ = self.params_[0] if want_intercept else 0.0
        return self


class GLM(BaseEstimator, RegressorMixin, _LinearInferenceMixin):
    """Generalized linear model by iteratively reweighted least squares (IRLS).

    THE UNIFICATION
    ---------------
    OLS assumes the response is Gaussian and the mean is linear in the features.
    A GLM relaxes both through two choices: a distribution FAMILY (Gaussian,
    Binomial, Poisson, Gamma) and a LINK function ``g`` connecting the linear
    predictor to the mean, ``g(mu) = X beta``. Logistic regression, Poisson
    regression and ordinary regression are then one algorithm with different
    (family, link) pairs -- which is the insight GLMs are famous for.

    WHY IRLS
    --------
    Maximising the likelihood has no closed form for most families, but the
    Newton step turns out to be a WEIGHTED least-squares fit on a linearised
    "working response" -- and the weights and response are recomputed each
    iteration from the current fit. So the whole family is solved by repeating an
    OLS solve, which is why one short loop covers logistic, Poisson and the rest.
    The final iteration's weighted ``(X'WX)^-1`` is the coefficient covariance,
    giving the standard errors for free.

    ``family`` in {"gaussian", "binomial", "poisson", "gamma"}.
    """

    def __init__(self, family="gaussian", fit_intercept=True, max_iter=100,
                 tol=1e-8):
        self.family = family
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    # each family: inverse-link (mean from eta), variance function, and the
    # deviance contribution used for the likelihood
    def _mean(self, eta):
        if self.family == "gaussian":
            return eta
        if self.family == "binomial":
            return 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        if self.family in ("poisson", "gamma"):
            return np.exp(np.clip(eta, -30, 30))
        raise ValueError(f"unknown family {self.family!r}")

    def _variance(self, mu):
        if self.family == "gaussian":
            return np.ones_like(mu)
        if self.family == "binomial":
            return np.clip(mu * (1 - mu), 1e-10, None)
        if self.family == "poisson":
            return np.clip(mu, 1e-10, None)
        if self.family == "gamma":
            return mu ** 2

    def _deriv(self, mu):
        """d(eta)/d(mu) for the canonical link -- links the working response."""
        if self.family == "gaussian":
            return np.ones_like(mu)
        if self.family == "binomial":
            return 1 / np.clip(mu * (1 - mu), 1e-10, None)
        if self.family == "poisson":
            return 1 / np.clip(mu, 1e-10, None)
        if self.family == "gamma":
            return -1 / mu ** 2

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.nobs_ = len(y)
        Xd = _add_intercept(X) if self.fit_intercept else X
        beta = np.zeros(Xd.shape[1])
        eta = Xd @ beta

        for _ in range(self.max_iter):
            mu = self._mean(eta)
            deriv = self._deriv(mu)
            # the IRLS working response z and weights w -- a linearisation of the
            # link around the current mean
            w = 1 / (self._variance(mu) * deriv ** 2)
            z = eta + (y - mu) * deriv
            WX = Xd * w[:, None]
            XtWX_inv = np.linalg.pinv(Xd.T @ WX)
            beta_new = XtWX_inv @ (WX.T @ z)
            if np.max(np.abs(beta_new - beta)) < self.tol:
                beta = beta_new
                break
            beta = beta_new
            eta = Xd @ beta

        self.params_ = beta
        self.coef_ = beta[1:] if self.fit_intercept else beta
        self.intercept_ = beta[0] if self.fit_intercept else 0.0
        mu = self._mean(eta)
        self.mu_ = mu

        # the coefficient covariance is the final weighted (X'WX)^-1 -- the
        # inverse observed information at the MLE
        w = 1 / (self._variance(mu) * self._deriv(mu) ** 2)
        cov_beta = np.linalg.pinv(Xd.T @ (Xd * w[:, None]))
        self._glm_finalize(y, mu, cov_beta, Xd.shape[1])
        return self

    def _glm_finalize(self, y, mu, cov_beta, n_params):
        n = len(y)
        self.df_resid_ = n - n_params
        self.bse_ = np.sqrt(np.diag(cov_beta))
        # GLMs use the NORMAL (z) distribution for inference, being MLE-based,
        # rather than OLS's t-distribution
        self.tvalues_ = self.params_ / self.bse_
        self.pvalues_ = 2 * stats.norm.sf(np.abs(self.tvalues_))
        self.cov_params_ = cov_beta
        self.llf_ = self._loglik(y, mu)
        self.aic_ = 2 * n_params - 2 * self.llf_
        self.bic_ = n_params * np.log(n) - 2 * self.llf_
        self.deviance_ = -2 * self.llf_

    def _loglik(self, y, mu):
        if self.family == "gaussian":
            resid = y - mu
            s2 = (resid @ resid) / len(y)
            return -0.5 * len(y) * (np.log(2 * np.pi * s2) + 1)
        if self.family == "binomial":
            mu = np.clip(mu, 1e-10, 1 - 1e-10)
            return float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
        if self.family == "poisson":
            from scipy.special import gammaln
            mu = np.clip(mu, 1e-10, None)
            return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))
        if self.family == "gamma":
            return float(-np.sum((y - mu) ** 2))    # a crude proxy

    def conf_int(self, alpha=0.05):
        check_is_fitted(self, "params_")
        z = stats.norm.ppf(1 - alpha / 2)
        return np.column_stack([self.params_ - z * self.bse_,
                                self.params_ + z * self.bse_])

    def summary(self):
        check_is_fitted(self, "params_")
        ci = self.conf_int()
        names = self._param_names()
        header = (f"GLM({self.family})   n={self.nobs_}   "
                  f"llf={self.llf_:.2f}   AIC={self.aic_:.1f}   "
                  f"deviance={self.deviance_:.2f}")
        lines = [header, "-" * len(header),
                 f"{'term':<12}{'coef':>10}{'std err':>10}{'z':>8}"
                 f"{'P>|z|':>9}{'[0.025':>10}{'0.975]':>10}"]
        for i, nm in enumerate(names):
            lines.append(f"{nm:<12}{self.params_[i]:>10.4f}{self.bse_[i]:>10.4f}"
                         f"{self.tvalues_[i]:>8.2f}{self.pvalues_[i]:>9.3f}"
                         f"{ci[i, 0]:>10.4f}{ci[i, 1]:>10.4f}")
        return "\n".join(lines)

    def predict(self, X):
        check_is_fitted(self, "params_")
        Xd = _add_intercept(check_array(X)) if self.fit_intercept else check_array(X)
        return self._mean(Xd @ self.params_)


class Logit(GLM):
    """Logistic regression as a GLM -- with the inference OLS-style tools lack.

    ``linear_model.LogisticRegression`` predicts classes; ``Logit`` additionally
    reports each coefficient's standard error, z-statistic, p-value and CI, so you
    can say "this predictor's effect is significant" with a number behind it.
    ``exp(coef)`` is the ODDS RATIO -- the multiplicative change in the odds per
    unit of the feature -- which is how logistic coefficients are read in
    practice.
    """

    def __init__(self, fit_intercept=True, max_iter=100, tol=1e-8):
        super().__init__(family="binomial", fit_intercept=fit_intercept,
                         max_iter=max_iter, tol=tol)


class Poisson(GLM):
    """Poisson regression for COUNT data.

    Counts are non-negative integers whose variance grows with their mean --
    both facts a Gaussian model gets wrong, producing impossible negative
    predictions and mis-stated SEs. Poisson regression's log link keeps
    predictions positive and its variance function ties spread to the mean, so
    ``exp(coef)`` reads as the multiplicative change in the expected count. The
    natural model for event rates: accidents, arrivals, defects.
    """

    def __init__(self, fit_intercept=True, max_iter=100, tol=1e-8):
        super().__init__(family="poisson", fit_intercept=fit_intercept,
                         max_iter=max_iter, tol=tol)


class Probit(BaseEstimator, RegressorMixin, _LinearInferenceMixin):
    """Probit regression: binary choice through the NORMAL CDF instead of logistic.

    Logit and Probit answer the same question -- probability of a 1 -- with
    almost indistinguishable fits; they differ only in the S-curve used (logistic
    vs standard-normal CDF). Probit is the convention in econometrics because it
    drops out of a latent-variable story (a normal utility crossing a threshold),
    while logit dominates in ML for its clean odds-ratio interpretation. Fitted
    here by Newton-Raphson on the exact probit log-likelihood, with the
    information-matrix covariance for inference.
    """

    def __init__(self, fit_intercept=True, max_iter=100, tol=1e-8):
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        self.nobs_ = len(y)
        Xd = _add_intercept(X) if self.fit_intercept else X
        beta = np.zeros(Xd.shape[1])

        for _ in range(self.max_iter):
            eta = Xd @ beta
            Phi = np.clip(stats.norm.cdf(eta), 1e-10, 1 - 1e-10)
            phi = stats.norm.pdf(eta)
            # gradient and Hessian of the probit log-likelihood
            grad = Xd.T @ (phi * (y - Phi) / (Phi * (1 - Phi)))
            w = phi ** 2 / (Phi * (1 - Phi))
            H = Xd.T @ (w[:, None] * Xd)
            step = np.linalg.pinv(H) @ grad
            beta = beta + step
            if np.max(np.abs(step)) < self.tol:
                break

        self.params_ = beta
        self.coef_ = beta[1:] if self.fit_intercept else beta
        self.intercept_ = beta[0] if self.fit_intercept else 0.0
        eta = Xd @ beta
        Phi = np.clip(stats.norm.cdf(eta), 1e-10, 1 - 1e-10)
        phi = stats.norm.pdf(eta)
        w = phi ** 2 / (Phi * (1 - Phi))
        cov_beta = np.linalg.pinv(Xd.T @ (w[:, None] * Xd))

        n_params = Xd.shape[1]
        self.df_resid_ = len(y) - n_params
        self.bse_ = np.sqrt(np.diag(cov_beta))
        self.tvalues_ = self.params_ / self.bse_
        self.pvalues_ = 2 * stats.norm.sf(np.abs(self.tvalues_))
        self.cov_params_ = cov_beta
        self.llf_ = float(np.sum(y * np.log(Phi) + (1 - y) * np.log(1 - Phi)))
        self.aic_ = 2 * n_params - 2 * self.llf_
        self.bic_ = n_params * np.log(len(y)) - 2 * self.llf_
        return self

    def conf_int(self, alpha=0.05):
        z = stats.norm.ppf(1 - alpha / 2)
        return np.column_stack([self.params_ - z * self.bse_,
                                self.params_ + z * self.bse_])

    def predict(self, X):
        check_is_fitted(self, "params_")
        Xd = _add_intercept(check_array(X)) if self.fit_intercept else check_array(X)
        return stats.norm.cdf(Xd @ self.params_)


def anova_lm(*models):
    """Compare nested OLS models by an F-test on their residual sums of squares.

    "Is the bigger model worth its extra parameters?" Each added predictor cannot
    increase the residual sum of squares, so the big model always fits the
    training data at least as well -- the question is whether the improvement
    exceeds what noise alone would give. The F-test answers it: it scales the drop
    in residual SS by the extra degrees of freedom and asks how surprising that
    drop is under the null that the added terms are zero. A small p-value says the
    extra terms earn their place.

    Pass two or more fitted ``OLS`` models in increasing size.
    """
    rows = []
    prev = None
    for m in models:
        ss = float(m.resid_ @ m.resid_)
        row = {"df_resid": m.df_resid_, "ss_resid": ss, "F": np.nan, "p": np.nan}
        if prev is not None:
            df_diff = prev["df_resid"] - m.df_resid_    # extra parameters
            ss_diff = prev["ss_resid"] - ss             # improvement in fit
            if df_diff > 0:
                f = (ss_diff / df_diff) / (ss / m.df_resid_)
                row["F"] = f
                row["p"] = stats.f.sf(f, df_diff, m.df_resid_)
        rows.append(row)
        prev = row
    return rows


__all__ = ["OLS", "WLS", "GLM", "Logit", "Probit", "Poisson", "anova_lm"]
