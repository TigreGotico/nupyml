"""Diagnostic tests: is the inference you just did actually valid?

A regression's p-values and confidence intervals rest on assumptions -- that the
residuals are uncorrelated, equal-variance, and roughly normal, and that a time
series is stationary. When those fail, the point estimates may still be fine but
the STANDARD ERRORS are wrong, so the p-values lie. These tests check the
assumptions, so you fit, then check, then believe -- in that order.

Each returns a ``(statistic, p_value)`` pair (or the test's natural output), with
a docstring saying what the null hypothesis is -- because "p < 0.05" means
nothing until you know what you are rejecting.
"""
import numpy as np
from scipy import stats


def durbin_watson(resid):
    """Test residuals for first-order autocorrelation. Returns the DW statistic.

    ``DW ~ 2`` means no autocorrelation; ``DW -> 0`` means strong POSITIVE
    autocorrelation (each residual resembles the last -- the classic sign of a
    misspecified time-series model or an omitted trend); ``DW -> 4`` means
    negative. Autocorrelated residuals violate the independence assumption, so
    OLS standard errors are too small and the model looks more significant than
    it is. It is a statistic, not a p-value -- the bounds are read from tables --
    but 1.5-2.5 is the usual "fine" range.
    """
    resid = np.asarray(resid, float)
    diff = np.diff(resid)
    return float((diff @ diff) / (resid @ resid))


def ljung_box(resid, lags=10):
    """Test for autocorrelation up to ``lags`` jointly. Null: no autocorrelation.

    Where Durbin-Watson checks only lag 1, Ljung-Box sums squared
    autocorrelations across many lags into a single chi-square statistic -- the
    standard check that a fitted time-series model's residuals are white noise
    (any structure left in them is signal the model missed). A small p-value
    rejects "the residuals are uncorrelated".
    """
    resid = np.asarray(resid, float)
    n = len(resid)
    resid = resid - resid.mean()
    denom = resid @ resid
    stat = 0.0
    for k in range(1, lags + 1):
        rk = (resid[:-k] @ resid[k:]) / denom
        stat += rk ** 2 / (n - k)
    stat *= n * (n + 2)
    p = stats.chi2.sf(stat, lags)
    return float(stat), float(p)


def breusch_pagan(resid, X):
    """Test for heteroskedasticity. Null: the error variance is CONSTANT.

    Regress the squared residuals on the predictors: if the spread of the errors
    depends on the features, that regression explains something, and the test's
    ``n * R^2`` statistic is large. Rejecting the null means the errors are
    heteroskedastic, so the classical OLS standard errors are invalid and you
    should switch to robust (HC) ones -- which is precisely the situation
    ``OLS(cov_type="HC3")`` exists for.
    """
    resid = np.asarray(resid, float)
    X = np.asarray(X, float)
    Xd = np.column_stack([np.ones(len(X)), X])
    r2_scaled = resid ** 2 / np.mean(resid ** 2)     # auxiliary response
    beta = np.linalg.pinv(Xd.T @ Xd) @ Xd.T @ r2_scaled
    fitted = Xd @ beta
    ss_tot = np.sum((r2_scaled - r2_scaled.mean()) ** 2)
    ss_expl = np.sum((fitted - r2_scaled.mean()) ** 2)
    stat = 0.5 * ss_expl                             # the LM statistic
    p = stats.chi2.sf(stat, X.shape[1])
    return float(stat), float(p)


def white_test(resid, X):
    """Heteroskedasticity test that also catches NONLINEAR variance patterns.

    Breusch-Pagan regresses squared residuals on the features linearly; White
    adds their squares and cross-products, so it detects variance that depends on
    the features in a curved or interacting way, not just linearly. More general,
    but it spends degrees of freedom fast as the feature count grows. Null: the
    variance is constant (homoskedastic).
    """
    resid = np.asarray(resid, float)
    X = np.asarray(X, float)
    n, p = X.shape
    # augment with squares and cross-products -- the "curved variance" terms
    cols = [np.ones(n), X]
    for i in range(p):
        for j in range(i, p):
            cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
    Z = np.column_stack(cols)
    r2 = resid ** 2
    beta = np.linalg.pinv(Z.T @ Z) @ Z.T @ r2
    fitted = Z @ beta
    ss_tot = np.sum((r2 - r2.mean()) ** 2)
    ss_res = np.sum((r2 - fitted) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    stat = n * r_squared                             # n * R^2 ~ chi2
    df = Z.shape[1] - 1
    return float(stat), float(stats.chi2.sf(stat, df))


def jarque_bera(x):
    """Test for normality via skewness and kurtosis. Null: the data is normal.

    A normal distribution has zero skew and a kurtosis of 3. Jarque-Bera combines
    the sample skewness and excess kurtosis into one chi-square statistic, so a
    small p-value says the data is non-normal -- too skewed, too heavy-tailed, or
    both. Used on regression residuals to check the normality that small-sample
    t- and F-tests assume (large samples lean on the CLT instead, so this matters
    most when n is small).
    """
    x = np.asarray(x, float)
    n = len(x)
    x = x - x.mean()
    s2 = np.mean(x ** 2)
    skew = np.mean(x ** 3) / s2 ** 1.5
    kurt = np.mean(x ** 4) / s2 ** 2
    stat = n / 6 * (skew ** 2 + (kurt - 3) ** 2 / 4)
    return float(stat), float(stats.chi2.sf(stat, 2))


def adf_test(x, max_lag=None):
    """Augmented Dickey-Fuller test for a UNIT ROOT. Null: the series is NON-stationary.

    Stationarity -- a constant mean and variance over time -- is required by AR/
    ARMA models, so this is the standard first check before fitting one. ADF
    regresses the change ``Δy_t`` on the level ``y_{t-1}`` (plus lagged changes):
    if the coefficient on the level is significantly negative, the series
    mean-reverts and is stationary. A SMALL p-value REJECTS the unit root, i.e.
    concludes the series IS stationary -- the opposite polarity to KPSS, which is
    why the two are run together.

    Returns ``(adf_statistic, approximate_p_value)``.
    """
    x = np.asarray(x, float)
    n = len(x)
    if max_lag is None:
        max_lag = int(np.ceil(12 * (n / 100) ** 0.25))   # Schwert's rule
        max_lag = min(max_lag, n // 2 - 2)
    dx = np.diff(x)
    lagged_level = x[:-1]
    # build the augmented regression: Δy_t on y_{t-1} and lagged Δy
    rows = max_lag
    Y = dx[rows:]
    cols = [np.ones(len(Y)), lagged_level[rows:]]
    for lag in range(1, max_lag + 1):
        cols.append(dx[rows - lag:-lag])
    Z = np.column_stack(cols)
    beta = np.linalg.pinv(Z.T @ Z) @ Z.T @ Y
    resid = Y - Z @ beta
    se = np.sqrt((resid @ resid) / (len(Y) - Z.shape[1])
                 * np.diag(np.linalg.pinv(Z.T @ Z)))
    adf_stat = beta[1] / se[1]                        # t-stat on the level term
    # interpolate an approximate p-value from MacKinnon's critical values
    crit = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}
    if adf_stat <= crit[0.01]:
        p = 0.01
    elif adf_stat <= crit[0.05]:
        p = 0.05
    elif adf_stat <= crit[0.10]:
        p = 0.10
    else:
        p = 0.5
    return float(adf_stat), float(p)


def kpss_test(x, regression="c"):
    """KPSS test. Null: the series IS stationary (the reverse of ADF's null).

    ADF and KPSS have OPPOSITE nulls, and running both is the standard practice --
    they cross-check each other. If ADF rejects (stationary) AND KPSS fails to
    reject (stationary), you are confident it is stationary; if they disagree, the
    evidence is weak and the series may be near a unit root or trend-stationary.
    KPSS builds its statistic from the cumulative sum of residuals: if the series
    is stationary those partial sums stay bounded, and a large statistic
    (unbounded drift) rejects stationarity.

    Returns ``(kpss_statistic, approximate_p_value)``.
    """
    x = np.asarray(x, float)
    n = len(x)
    if regression == "ct":
        t = np.arange(n)
        Z = np.column_stack([np.ones(n), t])
    else:
        Z = np.ones((n, 1))
    beta = np.linalg.pinv(Z.T @ Z) @ Z.T @ x
    resid = x - Z @ beta
    S = np.cumsum(resid)                              # partial sums of residuals
    # long-run variance estimate (Newey-West with a simple bandwidth)
    lag = int(4 * (n / 100) ** 0.25)
    s2 = (resid @ resid) / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        s2 += 2 * w * (resid[:-k] @ resid[k:]) / n
    stat = np.sum(S ** 2) / (n ** 2 * s2)
    crit = {0.10: 0.347, 0.05: 0.463, 0.01: 0.739}
    if stat >= crit[0.01]:
        p = 0.01
    elif stat >= crit[0.05]:
        p = 0.05
    elif stat >= crit[0.10]:
        p = 0.10
    else:
        p = 0.5
    return float(stat), float(p)


def granger_causality(y, x, max_lag=4):
    """Does the past of ``x`` help predict ``y``? Null: x does NOT Granger-cause y.

    "Causality" here is strictly predictive: x Granger-causes y if adding lagged
    values of x to a model that already contains lagged y significantly improves
    the prediction of y. It is NOT causation in the interventional sense (see the
    ``causal`` module for that) -- only "x's history carries information about y's
    future beyond y's own history". The test is an F-test comparing the restricted
    model (lags of y only) to the unrestricted one (lags of y AND x); a small
    p-value says x has predictive content for y.

    Returns ``(F_statistic, p_value)`` for the given ``max_lag``.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = len(y)
    L = max_lag
    Y = y[L:]
    # restricted design: lags of y only
    y_lags = np.column_stack([y[L - k:n - k] for k in range(1, L + 1)])
    Zr = np.column_stack([np.ones(len(Y)), y_lags])
    # unrestricted: add lags of x
    x_lags = np.column_stack([x[L - k:n - k] for k in range(1, L + 1)])
    Zu = np.column_stack([Zr, x_lags])

    def _rss(Z):
        beta = np.linalg.pinv(Z.T @ Z) @ Z.T @ Y
        r = Y - Z @ beta
        return float(r @ r)

    rss_r, rss_u = _rss(Zr), _rss(Zu)
    df1 = L                                           # the extra x-lag terms
    df2 = len(Y) - Zu.shape[1]
    f = ((rss_r - rss_u) / df1) / (rss_u / df2)
    return float(f), float(stats.f.sf(f, df1, df2))


__all__ = ["durbin_watson", "ljung_box", "breusch_pagan", "white_test",
           "jarque_bera", "adf_test", "kpss_test", "granger_causality"]
