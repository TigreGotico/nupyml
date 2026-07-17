"""More preprocessing transformers: reshaping distributions, encoding, and glue.

The scalers in ``preprocessing/__init__.py`` shift and scale. These reshape the
DISTRIBUTION itself, encode categories using the target, or wrap arbitrary
functions into the transformer contract.
"""
import numpy as np

from ..base import BaseEstimator, TransformerMixin, check_is_fitted
from ..utils import check_array, check_random_state


class PowerTransformer(BaseEstimator, TransformerMixin):
    """Make skewed data more GAUSSIAN, by learning a power to apply.

    WHY
    ---
    Many models -- linear regression, LDA, anything assuming normal residuals --
    work best on roughly symmetric, Gaussian-ish features. Real data is often
    skewed (incomes, counts, concentrations). A power transform finds the exponent
    that best symmetrises each feature.

    * Yeo-Johnson (the default) handles NEGATIVE values; Box-Cox needs strictly
      positive input. That is the only practical difference, and why Yeo-Johnson
      is the safer default.

    The exponent ``lambda`` is chosen per feature by MAXIMUM LIKELIHOOD -- the
    value that makes the transformed data most Gaussian. So it is not a fixed
    "take the log"; it is a log, a square root, a reciprocal, or anything between,
    picked to fit the actual skew. This is a principled generalisation of the
    ad-hoc "just log-transform it" that people reach for.

    Yeo & Johnson (2000); Box & Cox (1964).
    """

    def __init__(self, method="yeo-johnson", standardize=True):
        self.method = method
        self.standardize = standardize

    def _yeo_johnson(self, x, lam):
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        # four cases: the transform is defined piecewise to stay smooth at lam=0
        # and to handle negatives, which is exactly what Box-Cox could not
        if abs(lam) < 1e-6:
            out[pos] = np.log1p(x[pos])
        else:
            out[pos] = ((x[pos] + 1) ** lam - 1) / lam
        if abs(lam - 2) < 1e-6:
            out[~pos] = -np.log1p(-x[~pos])
        else:
            out[~pos] = -(((-x[~pos] + 1) ** (2 - lam) - 1) / (2 - lam))
        return out

    def _neg_loglik(self, lam, x):
        t = self._yeo_johnson(x, lam)
        var = np.var(t) + 1e-12
        # the profile log-likelihood of a Gaussian after the transform, plus the
        # Jacobian term that rewards spreading the data out
        jac = np.sum(np.sign(x) * np.log1p(np.abs(x)))
        return 0.5 * len(x) * np.log(var) - (lam - 1) * jac

    def fit(self, X, y=None):
        from scipy.optimize import minimize_scalar
        X = check_array(X)
        self.lambdas_ = np.empty(X.shape[1])
        for j in range(X.shape[1]):
            # find the lambda that maximises Gaussian likelihood for this feature
            res = minimize_scalar(self._neg_loglik, bounds=(-2, 3),
                                  args=(X[:, j],), method="bounded")
            self.lambdas_[j] = res.x
        Xt = self._transform_raw(X)
        self._mean = Xt.mean(axis=0)
        self._std = Xt.std(axis=0)
        self._std[self._std == 0] = 1.0
        return self

    def _transform_raw(self, X):
        return np.column_stack([self._yeo_johnson(X[:, j], self.lambdas_[j])
                                for j in range(X.shape[1])])

    def transform(self, X):
        check_is_fitted(self, "lambdas_")
        Xt = self._transform_raw(check_array(X))
        if self.standardize:
            Xt = (Xt - self._mean) / self._std      # zero-mean unit-variance after
        return Xt


class QuantileTransformer(BaseEstimator, TransformerMixin):
    """Force any distribution into uniform or Gaussian, by rank.

    THE IDEA
    --------
    Map each value to its QUANTILE (its rank fraction in the training data), which
    is uniform on [0, 1] by construction; optionally push that through the inverse
    Gaussian CDF to get a normal distribution. Unlike the power transform, this
    makes NO assumption about the shape -- it works for any distribution, however
    weird, multi-modal or heavy-tailed.

    THE PRICE
    ---------
    It is a non-parametric, rank-based transform, so it is ROBUST to outliers (an
    extreme value just becomes "the highest rank") but it DISTORTS distances and
    can wash out fine structure between nearby values -- a monotone but non-linear
    warping that is not invertible outside the observed range. Powerful when you
    need a specific output shape and do not care about preserving the exact
    geometry; the power transform is gentler when you do.
    """

    def __init__(self, n_quantiles=1000, output_distribution="uniform"):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution

    def fit(self, X, y=None):
        X = check_array(X)
        n_q = min(self.n_quantiles, len(X))
        self.quantiles_ = np.percentile(X, np.linspace(0, 100, n_q), axis=0)
        self.references_ = np.linspace(0, 1, n_q)
        return self

    def transform(self, X):
        from scipy.stats import norm
        check_is_fitted(self, "quantiles_")
        X = check_array(X)
        out = np.empty_like(X, dtype=float)
        for j in range(X.shape[1]):
            # interpolate each value to its rank fraction in the training data
            out[:, j] = np.interp(X[:, j], self.quantiles_[:, j], self.references_)
        if self.output_distribution == "normal":
            out = norm.ppf(np.clip(out, 1e-7, 1 - 1e-7))   # uniform -> Gaussian
        return out


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Encode a category by the mean target within it, smoothed toward the prior.

    THE IDEA AND ITS DANGER
    -----------------------
    One-hot encoding a high-cardinality category (zip code, product id) explodes
    the feature space. Target encoding replaces each category with the mean of the
    target for that category -- one column, however many categories. But the naive
    version LEAKS: a category seen once gets encoded as exactly that row's label,
    handing the model the answer (the same leak ``ordered_target_statistic`` in
    ``ensemble`` fixes with ordering).

    THE SMOOTHING FIX
    -----------------
    This version blends each category's mean toward the GLOBAL mean, weighted by
    how many times the category appears::

        encoding = (count * category_mean + smoothing * global_mean)
                   / (count + smoothing)

    A frequent category trusts its own mean; a rare one falls back to the global
    prior, so a singleton is NOT encoded as its own label. Smoothing bounds the
    leak rather than eliminating it -- for the strict version, cross-fit or use
    the ordered statistic. This is the standard, practical target encoder.
    """

    def __init__(self, smoothing=10.0):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = check_array(X, dtype=object)
        y = np.asarray(y, dtype=float)
        self.global_mean_ = y.mean()
        self.encodings_ = []
        for j in range(X.shape[1]):
            enc = {}
            for cat in np.unique(X[:, j]):
                mask = X[:, j] == cat
                count = mask.sum()
                cat_mean = y[mask].mean()
                # shrink toward the prior by count -- rare categories trust it more
                enc[cat] = ((count * cat_mean + self.smoothing * self.global_mean_)
                            / (count + self.smoothing))
            self.encodings_.append(enc)
        return self

    def transform(self, X):
        check_is_fitted(self, "encodings_")
        X = check_array(X, dtype=object)
        out = np.empty(X.shape, dtype=float)
        for j in range(X.shape[1]):
            enc = self.encodings_[j]
            # unseen categories fall back to the global mean, never crash
            out[:, j] = [enc.get(v, self.global_mean_) for v in X[:, j]]
        return out


class FunctionTransformer(BaseEstimator, TransformerMixin):
    """Wrap an arbitrary function into the transformer contract.

    The glue that lets any Python function -- ``np.log1p``, a custom clip, a
    reshape -- live inside a ``Pipeline`` alongside real estimators, with ``fit``
    /``transform`` and (optionally) ``inverse_transform``. It learns nothing; it
    exists so a fixed, stateless step does not have to be a bespoke class. The
    honest note: because it holds no state, putting data-dependent logic in the
    function (like standardising by the batch's own mean) silently leaks between
    train and test -- keep the function truly stateless.
    """

    def __init__(self, func=None, inverse_func=None):
        self.func = func
        self.inverse_func = inverse_func

    def fit(self, X, y=None):
        check_array(X)
        return self

    def transform(self, X):
        X = check_array(X)
        return X if self.func is None else self.func(X)

    def inverse_transform(self, X):
        if self.inverse_func is None:
            return np.asarray(X)
        return self.inverse_func(np.asarray(X))


__all__ = ["PowerTransformer", "QuantileTransformer", "TargetEncoder",
           "FunctionTransformer"]
