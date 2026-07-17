"""Statistical inference: standard errors, tests, and intervals.

Validated against CLOSED-FORM results and scipy.stats -- the mathematical ground
truth -- rather than another library, so the checks are exact where the formula
is exact (OLS covariance, t-to-p conversion) and property-based where it is not
(coefficient recovery, CI coverage, diagnostic polarity).
"""
import numpy as np
import pytest
from scipy import stats as sps

from nupyml.stats import OLS, WLS, GLM, Logit, Probit, Poisson, anova_lm
from nupyml.stats import (durbin_watson, ljung_box, breusch_pagan, white_test,
                          jarque_bera, adf_test, kpss_test, granger_causality)


# --- OLS inference --------------------------------------------------------

@pytest.fixture
def ols_data():
    rng = np.random.RandomState(0)
    n = 300
    X = rng.normal(size=(n, 3))
    y = 5 + X @ [2.0, -1.0, 0.5] + rng.normal(0, 1.0, n)
    return X, y


def test_ols_recovers_coefficients(ols_data):
    X, y = ols_data
    m = OLS().fit(X, y)
    # intercept 5, then slopes -- recovered within a few standard errors
    assert m.params_[0] == pytest.approx(5.0, abs=0.2)
    assert np.allclose(m.params_[1:], [2.0, -1.0, 0.5], atol=0.15)


def test_ols_standard_errors_match_the_closed_form(ols_data):
    """SE = sqrt(diag(sigma^2 (X'X)^-1)) exactly, with the n-p correction."""
    X, y = ols_data
    m = OLS().fit(X, y)
    Xd = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.solve(Xd.T @ Xd, Xd.T @ y)
    resid = y - Xd @ beta
    sigma2 = (resid @ resid) / (len(y) - Xd.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(Xd.T @ Xd)))
    assert np.allclose(m.bse_, se, rtol=1e-8)


def test_ols_pvalues_match_scipy_t_distribution(ols_data):
    """p-value is the two-sided tail of the t-distribution at the t-statistic."""
    X, y = ols_data
    m = OLS().fit(X, y)
    expected = 2 * sps.t.sf(np.abs(m.tvalues_), m.df_resid_)
    assert np.allclose(m.pvalues_, expected, rtol=1e-10)


def test_ols_rsquared_matches_manual(ols_data):
    X, y = ols_data
    m = OLS().fit(X, y)
    pred = m.predict(X)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    assert m.rsquared_ == pytest.approx(1 - ss_res / ss_tot, rel=1e-10)


def test_ols_confidence_intervals_cover_at_the_stated_rate():
    """A 95% CI should contain the true coefficient ~95% of the time across
    resamples -- the defining property of a confidence interval."""
    rng = np.random.RandomState(0)
    covered = 0
    trials = 200
    for _ in range(trials):
        X = rng.normal(size=(120, 2))
        y = 1.0 + X @ [3.0, -2.0] + rng.normal(0, 1.0, 120)
        m = OLS().fit(X, y)
        ci = m.conf_int(alpha=0.05)
        # check the first slope (param index 1)
        if ci[1, 0] <= 3.0 <= ci[1, 1]:
            covered += 1
    rate = covered / trials
    assert 0.90 < rate < 0.99          # ~0.95, allowing sampling slack


def test_ols_significant_effect_has_small_pvalue(ols_data):
    """A genuine effect should be flagged significant; pure noise should not."""
    X, y = ols_data
    # append a pure-noise column that does not affect y
    X2 = np.column_stack([X, np.random.RandomState(1).normal(size=len(X))])
    m = OLS().fit(X2, y)
    assert np.all(m.pvalues_[1:4] < 0.01)     # the three real slopes
    assert m.pvalues_[4] > 0.05               # the noise column


def test_ols_f_test_rejects_the_null_of_no_relationship(ols_data):
    X, y = ols_data
    m = OLS().fit(X, y)
    assert m.f_pvalue_ < 1e-10                # the model explains real variance


def test_robust_standard_errors_differ_under_heteroskedasticity():
    """When the error variance grows with x, HC3 SEs should diverge from the
    classical ones -- which is the whole reason to use them."""
    rng = np.random.RandomState(0)
    x = rng.uniform(0, 5, 400)
    y = 1 + 2 * x + rng.normal(0, 1, 400) * x    # variance grows with x
    X = x.reshape(-1, 1)
    classical = OLS(cov_type="nonrobust").fit(X, y).bse_
    robust = OLS(cov_type="HC3").fit(X, y).bse_
    assert not np.allclose(classical, robust, rtol=0.05)


def test_ols_summary_is_a_string(ols_data):
    X, y = ols_data
    s = OLS().fit(X, y).summary()
    assert "coef" in s and "P>|t|" in s and "const" in s


def test_wls_is_more_efficient_than_ols_under_heteroskedasticity():
    """WLS with correct inverse-variance weights is more EFFICIENT than OLS when
    the noise is heteroskedastic -- a property of the sampling distribution, so it
    shows up in the AVERAGE error over resamples, not necessarily any one draw."""
    rng = np.random.RandomState(0)
    ols_errs, wls_errs = [], []
    for _ in range(100):
        x = rng.uniform(0, 5, 400)
        noise_sd = 0.2 + x                     # known, growing noise
        y = 1 + 2 * x + rng.normal(0, 1, 400) * noise_sd
        X = x.reshape(-1, 1)
        ols_errs.append(abs(OLS().fit(X, y).coef_[0] - 2.0))
        wls_errs.append(
            abs(WLS().fit(X, y, sample_weight=1 / noise_sd ** 2).coef_[0] - 2.0))
    assert np.mean(wls_errs) < np.mean(ols_errs)


def test_wls_fits_an_exact_line():
    """The intercept-whitening regression fix: a perfectly linear, weighted fit
    must recover the line exactly (this is what broke when only the features,
    not the intercept column, were whitened)."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])          # y = 2x + 1
    w = WLS().fit(X, y, sample_weight=[1.0, 2.0, 3.0, 4.0])
    assert w.coef_[0] == pytest.approx(2.0, abs=1e-6)
    assert w.intercept_ == pytest.approx(1.0, abs=1e-6)


# --- GLM / discrete choice ------------------------------------------------

def test_logit_recovers_coefficients_and_flags_significance():
    rng = np.random.RandomState(0)
    n = 800
    X = rng.normal(size=(n, 2))
    p = 1 / (1 + np.exp(-(X @ [1.5, -2.0])))
    y = (rng.uniform(size=n) < p).astype(float)
    m = Logit().fit(X, y)
    assert np.allclose(m.coef_, [1.5, -2.0], atol=0.4)
    assert np.all(m.pvalues_[1:] < 0.01)       # both effects are real
    assert np.all(m.bse_ > 0)


def test_logit_odds_ratio_interpretation():
    """exp(coef) is the odds ratio; a positive coef means >1 odds ratio."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(600, 1))
    y = (rng.uniform(size=600) < 1 / (1 + np.exp(-(1.0 * X[:, 0])))).astype(float)
    m = Logit().fit(X, y)
    assert np.exp(m.coef_[0]) > 1.0


def test_poisson_recovers_rate_coefficients():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(1000, 2))
    rate = np.exp(0.5 + X @ [0.3, -0.2])
    y = rng.poisson(rate).astype(float)
    m = Poisson().fit(X, y)
    assert m.params_[0] == pytest.approx(0.5, abs=0.15)
    assert np.allclose(m.coef_, [0.3, -0.2], atol=0.1)


def test_probit_and_logit_agree_on_direction():
    """Probit and logit fit near-identically up to a scale factor; the signs and
    significance must match."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(800, 2))
    y = (rng.uniform(size=800) < 1 / (1 + np.exp(-(X @ [1.0, -1.5])))).astype(float)
    logit = Logit().fit(X, y)
    probit = Probit().fit(X, y)
    assert np.all(np.sign(logit.coef_) == np.sign(probit.coef_))
    # logit coefficients are ~1.6x probit's (the standard scale relationship)
    ratio = logit.coef_ / probit.coef_
    assert np.all((ratio > 1.3) & (ratio < 2.0))


def test_glm_gaussian_matches_ols():
    """A Gaussian-family GLM with identity link IS ordinary least squares."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 2))
    y = 1 + X @ [2.0, -1.0] + rng.normal(0, 1, 200)
    ols = OLS().fit(X, y)
    glm = GLM(family="gaussian").fit(X, y)
    assert np.allclose(ols.params_, glm.params_, atol=1e-6)


def test_glm_aic_penalises_extra_parameters():
    """AIC should prefer the true model over one padded with noise features."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(400, 2))
    y = (rng.uniform(size=400) < 1 / (1 + np.exp(-(X @ [1.0, -1.0])))).astype(float)
    small = Logit().fit(X, y)
    big = Logit().fit(np.column_stack([X, rng.normal(size=(400, 3))]), y)
    assert small.aic_ < big.aic_


def test_anova_detects_useful_added_features(ols_data):
    """The F-test should reject 'the extra features are useless' when they help."""
    X, y = ols_data
    m1 = OLS().fit(X[:, :1], y)
    m2 = OLS().fit(X, y)
    result = anova_lm(m1, m2)
    assert result[1]["p"] < 1e-6              # adding x1, x2 significantly helps


# --- diagnostics ----------------------------------------------------------

def test_durbin_watson_near_two_for_white_noise():
    resid = np.random.RandomState(0).normal(size=500)
    assert 1.8 < durbin_watson(resid) < 2.2


def test_durbin_watson_low_for_positive_autocorrelation():
    """A series where each value tracks the last has DW well below 2."""
    rng = np.random.RandomState(0)
    resid = np.zeros(500)
    for t in range(1, 500):
        resid[t] = 0.8 * resid[t - 1] + rng.normal()
    assert durbin_watson(resid) < 1.0


def test_ljung_box_rejects_autocorrelated_residuals():
    rng = np.random.RandomState(0)
    ar = np.zeros(500)
    for t in range(1, 500):
        ar[t] = 0.7 * ar[t - 1] + rng.normal()
    white = rng.normal(size=500)
    assert ljung_box(ar)[1] < 0.01           # rejects: correlated
    assert ljung_box(white)[1] > 0.05        # does not reject: white noise


def test_breusch_pagan_detects_heteroskedasticity():
    rng = np.random.RandomState(0)
    x = rng.uniform(0, 5, 400)
    homo = rng.normal(0, 1, 400)
    hetero = rng.normal(0, 1, 400) * x        # variance grows with x
    X = x.reshape(-1, 1)
    assert breusch_pagan(hetero, X)[1] < 0.05     # rejects constant variance
    assert breusch_pagan(homo, X)[1] > 0.05       # constant variance ok


def test_white_test_detects_heteroskedasticity():
    rng = np.random.RandomState(0)
    x = rng.uniform(0, 5, 400)
    hetero = rng.normal(0, 1, 400) * x
    assert white_test(hetero, x.reshape(-1, 1))[1] < 0.05


def test_jarque_bera_rejects_non_normal():
    rng = np.random.RandomState(0)
    normal = rng.normal(size=1000)
    skewed = rng.exponential(size=1000)      # very non-normal
    assert jarque_bera(normal)[1] > 0.05     # does not reject normality
    assert jarque_bera(skewed)[1] < 0.01     # rejects normality


def test_adf_and_kpss_have_opposite_polarity():
    """The two stationarity tests cross-check: on a stationary series ADF rejects
    (its null is a unit root) while KPSS does not (its null is stationarity), and
    on a random walk they flip. Both agreeing is the confident verdict."""
    # a longer series so each single-sample test statistic is stable (KPSS on a
    # short white-noise series is noisy enough to occasionally cross its 10%
    # critical value by chance)
    rng = np.random.RandomState(1)
    stationary = rng.normal(0, 1, 800)
    random_walk = np.cumsum(rng.normal(0, 1, 800))

    assert adf_test(stationary)[1] <= 0.05        # ADF: reject unit root
    assert kpss_test(stationary)[1] >= 0.10       # KPSS: keep stationarity
    assert adf_test(random_walk)[1] >= 0.10       # ADF: keep unit root
    assert kpss_test(random_walk)[1] <= 0.05      # KPSS: reject stationarity


def test_granger_causality_detects_the_driving_direction():
    """x drives y with a lag, so x Granger-causes y strongly; the reverse is not
    significant at a strict threshold."""
    rng = np.random.RandomState(0)
    n = 500
    x = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.4 * y[t - 1] + 0.8 * x[t - 1] + rng.normal(0, 0.5)
    assert granger_causality(y, x)[1] < 0.001     # x -> y: strong
    assert granger_causality(x, y)[1] > 0.001      # y -> x: much weaker
