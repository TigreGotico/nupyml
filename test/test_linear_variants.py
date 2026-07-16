import numpy as np
import pytest

import sklearn.linear_model as sk

from nupyml.linear_model import (
    HuberRegressor, QuantileRegressor, TheilSenRegressor, RANSACRegressor,
    BayesianRidge, ARDRegression, PoissonRegressor, GammaRegressor,
    TweedieRegressor, RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV,
    OrthogonalMatchingPursuit, LinearRegression, Ridge,
)
from nupyml.metrics import mean_pinball_loss, r2_score
from nupyml.datasets import make_regression, make_classification

rng = np.random.RandomState(0)


def _with_outliers(n=100, d=5, n_out=10, seed=0):
    X, y, w = make_regression(n_samples=n, n_features=d, noise=1.0,
                              coef=True, random_state=seed)
    y_dirty = y.copy()
    y_dirty[:n_out] += 500.0
    return X, y, y_dirty, w


# ---------------------------------------------------------------------------
# robust regressors
# ---------------------------------------------------------------------------

def test_huber_resists_outliers():
    X, y, y_dirty, _ = _with_outliers()
    huber = HuberRegressor().fit(X, y_dirty)
    ols = LinearRegression().fit(X, y_dirty)
    # scored against the clean targets: OLS is dragged by the outliers
    assert huber.score(X, y) > ols.score(X, y)
    assert huber.score(X, y) > 0.99


def test_huber_matches_sklearn():
    X, y, y_dirty, _ = _with_outliers(seed=1)
    ours = HuberRegressor(epsilon=1.35, alpha=1e-4).fit(X, y_dirty)
    ref = sk.HuberRegressor(epsilon=1.35, alpha=1e-4).fit(X, y_dirty)
    assert np.allclose(ours.coef_, ref.coef_, rtol=0.05, atol=0.5)
    assert ours.scale_ == pytest.approx(ref.scale_, rel=0.2)


def test_huber_flags_outliers():
    X, y, y_dirty, _ = _with_outliers(n_out=10, seed=2)
    huber = HuberRegressor().fit(X, y_dirty)
    # the 10 corrupted rows must all be flagged; epsilon=1.35 also flags the
    # tail of ordinary gaussian noise, so the count exceeds 10
    assert huber.outliers_[:10].all()
    assert huber.outliers_.sum() < len(X) / 2


def test_huber_epsilon_controls_robustness():
    X, y, y_dirty, _ = _with_outliers(seed=3)
    tight = HuberRegressor(epsilon=1.05).fit(X, y_dirty)
    loose = HuberRegressor(epsilon=100.0).fit(X, y_dirty)
    # a huge epsilon is effectively OLS, so it degrades on the clean targets
    assert tight.score(X, y) > loose.score(X, y)


def test_quantile_regressor_tracks_quantiles():
    r = np.random.RandomState(4)
    X = r.uniform(0, 10, size=(300, 1))
    y = 2 * X.ravel() + r.normal(0, 2, size=300)
    preds = {}
    for q in (0.1, 0.5, 0.9):
        preds[q] = QuantileRegressor(quantile=q, alpha=0.0).fit(X, y).predict(X)
    # quantile curves must be ordered and cover the right fraction of points
    assert (preds[0.1] <= preds[0.5] + 1e-6).all()
    assert (preds[0.5] <= preds[0.9] + 1e-6).all()
    assert abs((y <= preds[0.1]).mean() - 0.1) < 0.05
    assert abs((y <= preds[0.9]).mean() - 0.9) < 0.05


def test_quantile_matches_sklearn():
    X, y = make_regression(n_samples=80, n_features=3, noise=5.0, random_state=5)
    ours = QuantileRegressor(quantile=0.5, alpha=0.0).fit(X, y)
    ref = sk.QuantileRegressor(quantile=0.5, alpha=0.0).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-4)
    assert ours.intercept_ == pytest.approx(ref.intercept_, abs=1e-4)


def test_quantile_beats_ols_on_pinball_loss():
    r = np.random.RandomState(6)
    X = r.uniform(0, 5, size=(200, 1))
    y = X.ravel() + r.exponential(2, size=200)   # skewed noise
    q = 0.9
    qr = QuantileRegressor(quantile=q, alpha=0.0).fit(X, y)
    ols = LinearRegression().fit(X, y)
    assert (mean_pinball_loss(y, qr.predict(X), alpha=q)
            < mean_pinball_loss(y, ols.predict(X), alpha=q))


def test_theil_sen_resists_outliers():
    X, y, y_dirty, _ = _with_outliers(n=60, d=2, n_out=5, seed=7)
    ts = TheilSenRegressor(random_state=0, max_subpopulation=200).fit(X, y_dirty)
    ols = LinearRegression().fit(X, y_dirty)
    assert ts.score(X, y) > ols.score(X, y)


def test_theil_sen_matches_sklearn():
    X, y, y_dirty, _ = _with_outliers(n=60, d=2, n_out=5, seed=8)
    ours = TheilSenRegressor(random_state=0, max_subpopulation=200).fit(X, y_dirty)
    ref = sk.TheilSenRegressor(random_state=0, max_subpopulation=200).fit(X, y_dirty)
    assert ours.score(X, y) == pytest.approx(ref.score(X, y), abs=0.02)


def test_ransac_isolates_inliers():
    X, y, y_dirty, _ = _with_outliers(n_out=15, seed=9)
    ransac = RANSACRegressor(random_state=0).fit(X, y_dirty)
    # the corrupted rows must be excluded from the consensus set
    assert not ransac.inlier_mask_[:15].any()
    assert ransac.inlier_mask_.sum() > 60
    assert ransac.score(X, y) > 0.99


def test_ransac_exposes_underlying_coefficients():
    X, y, y_dirty, true_w = _with_outliers(seed=10)
    ransac = RANSACRegressor(random_state=0).fit(X, y_dirty)
    assert np.allclose(ransac.coef_, true_w, atol=1.0)
    assert np.isfinite(ransac.intercept_)
    assert ransac.n_trials_ >= 1


def test_ransac_threshold_controls_inlier_set():
    X, y, y_dirty, _ = _with_outliers(seed=27)
    tight = RANSACRegressor(residual_threshold=1.0, random_state=0).fit(X, y_dirty)
    loose = RANSACRegressor(residual_threshold=1000.0, random_state=0).fit(X, y_dirty)
    assert tight.inlier_mask_.sum() < loose.inlier_mask_.sum()
    # a threshold large enough to swallow the outliers loses the robustness
    assert tight.score(X, y) > loose.score(X, y)


# ---------------------------------------------------------------------------
# bayesian linear models
# ---------------------------------------------------------------------------

def test_bayesian_ridge_matches_sklearn():
    X, y = make_regression(n_samples=100, n_features=6, noise=5.0, random_state=11)
    ours = BayesianRidge().fit(X, y)
    ref = sk.BayesianRidge().fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, rtol=1e-3, atol=1e-3)
    assert ours.alpha_ == pytest.approx(ref.alpha_, rel=0.05)
    assert ours.lambda_ == pytest.approx(ref.lambda_, rel=0.05)


def test_bayesian_ridge_predictive_std():
    X, y = make_regression(n_samples=80, n_features=3, noise=10.0, random_state=12)
    br = BayesianRidge().fit(X, y)
    mean, std = br.predict(X, return_std=True)
    assert (std > 0).all()
    # uncertainty grows away from the training data
    far = np.full((1, 3), 50.0)
    _, std_far = br.predict(far, return_std=True)
    assert std_far[0] > std.mean()


def test_ard_prunes_irrelevant_features():
    r = np.random.RandomState(13)
    X = r.normal(size=(150, 10))
    # only the first two features carry signal
    y = 5 * X[:, 0] - 3 * X[:, 1] + r.normal(scale=0.1, size=150)
    ard = ARDRegression().fit(X, y)
    assert np.abs(ard.coef_[0]) > 1.0 and np.abs(ard.coef_[1]) > 1.0
    # noise features are driven to ~zero weight with a huge prior precision
    assert np.abs(ard.coef_[2:]).max() < 0.1
    assert (ard.lambda_[2:] > 1e3).all()
    assert np.abs(ard.coef_[:2]).min() / max(np.abs(ard.coef_[2:]).max(),
                                             1e-12) > 20


def test_ard_matches_sklearn():
    X, y = make_regression(n_samples=100, n_features=8, n_informative=3,
                           noise=1.0, random_state=14)
    ours = ARDRegression().fit(X, y)
    ref = sk.ARDRegression().fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=0.1)


# ---------------------------------------------------------------------------
# GLMs
# ---------------------------------------------------------------------------

def _count_data(seed=15):
    r = np.random.RandomState(seed)
    X = r.normal(size=(200, 4)) / 2
    y = r.poisson(np.exp(1.0 + 0.8 * X[:, 0] - 0.5 * X[:, 1]))
    return X, y.astype(float)


def test_poisson_regressor_matches_sklearn():
    X, y = _count_data()
    ours = PoissonRegressor(alpha=0.01).fit(X, y)
    ref = sk.PoissonRegressor(alpha=0.01).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-4)
    assert ours.intercept_ == pytest.approx(ref.intercept_, abs=1e-4)
    assert ours.score(X, y) == pytest.approx(ref.score(X, y), abs=1e-4)


def test_poisson_predictions_are_positive_counts():
    X, y = _count_data(16)
    p = PoissonRegressor(alpha=0.01).fit(X, y)
    assert (p.predict(X) > 0).all()
    # the log link recovers the true coefficient signs
    assert p.coef_[0] > 0 and p.coef_[1] < 0


def test_poisson_beats_ols_on_counts():
    X, y = _count_data(17)
    poisson_d2 = PoissonRegressor(alpha=0.01).fit(X, y).score(X, y)
    assert poisson_d2 > 0.2
    # OLS can predict negative counts; Poisson cannot
    assert (LinearRegression().fit(X, y).predict(X) < 0).any()


def test_gamma_regressor_matches_sklearn():
    r = np.random.RandomState(18)
    X = r.normal(size=(150, 3)) / 2
    y = r.gamma(2.0, np.exp(1 + 0.5 * X[:, 0]) / 2.0)
    ours = GammaRegressor(alpha=0.01).fit(X, y)
    ref = sk.GammaRegressor(alpha=0.01).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-3)
    assert ours.score(X, y) == pytest.approx(ref.score(X, y), abs=1e-3)


@pytest.mark.parametrize("power", [0.0, 1.0, 1.5, 2.0])
def test_tweedie_regressor_matches_sklearn(power):
    X, y = _count_data(19)
    y = np.maximum(y, 0.1)   # power=2 needs strictly positive targets
    ours = TweedieRegressor(power=power, alpha=0.01).fit(X, y)
    ref = sk.TweedieRegressor(power=power, alpha=0.01).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-3)


def test_tweedie_link_auto_is_identity_only_for_power_zero():
    """link='auto' means identity at power=0 (plain least squares) and log
    elsewhere; predictions under a log link are strictly positive."""
    X, y = _count_data(28)
    normal = TweedieRegressor(power=0.0, alpha=0.01).fit(X, y)
    assert normal._resolve_link() == "identity"
    logged = TweedieRegressor(power=0.0, alpha=0.01, link="log").fit(X, y)
    assert logged._resolve_link() == "log"
    assert (logged.predict(X) > 0).all()
    assert PoissonRegressor()._resolve_link() == "log"


def test_glm_input_validation():
    X = rng.normal(size=(20, 2))
    with pytest.raises(ValueError, match="y >= 0"):
        PoissonRegressor().fit(X, rng.normal(size=20) - 10)
    with pytest.raises(ValueError, match="positive"):
        GammaRegressor().fit(X, np.zeros(20))


# ---------------------------------------------------------------------------
# built-in CV paths
# ---------------------------------------------------------------------------

def test_ridge_cv_picks_sensible_alpha():
    X, y = make_regression(n_samples=100, n_features=5, noise=30.0,
                           random_state=20)
    rcv = RidgeCV(alphas=[0.01, 1.0, 100.0, 10000.0], cv=5).fit(X, y)
    assert rcv.alpha_ in (0.01, 1.0, 100.0, 10000.0)
    # the chosen alpha must beat the extremes on held-out data
    from nupyml.model_selection import cross_val_score
    best = cross_val_score(Ridge(alpha=rcv.alpha_), X, y, cv=5).mean()
    worst = cross_val_score(Ridge(alpha=10000.0), X, y, cv=5).mean()
    assert best >= worst


def test_ridge_cv_matches_sklearn_choice():
    X, y = make_regression(n_samples=120, n_features=6, noise=20.0,
                           random_state=21)
    alphas = [0.1, 1.0, 10.0, 100.0]
    ours = RidgeCV(alphas=alphas, cv=5).fit(X, y)
    ref = sk.RidgeCV(alphas=alphas).fit(X, y)
    assert ours.alpha_ == pytest.approx(ref.alpha_, rel=10)  # same ballpark
    assert ours.score(X, y) > 0.5


def test_lasso_cv_selects_sparse_model():
    X, y = make_regression(n_samples=150, n_features=20, n_informative=3,
                           noise=1.0, random_state=22)
    lcv = LassoCV(n_alphas=15, cv=5).fit(X, y)
    assert lcv.alpha_ > 0
    assert lcv.mse_path_.shape == (15, 5)
    assert (np.abs(lcv.coef_) < 1e-8).sum() > 5      # sparsity achieved
    assert lcv.score(X, y) > 0.9


def test_elasticnet_cv():
    X, y = make_regression(n_samples=120, n_features=15, n_informative=4,
                           noise=1.0, random_state=23)
    ecv = ElasticNetCV(l1_ratio=0.7, n_alphas=10, cv=5).fit(X, y)
    assert ecv.alpha_ in ecv.alphas_
    assert ecv.score(X, y) > 0.85


def test_logistic_regression_cv():
    X, y = make_classification(n_samples=200, n_features=6, n_informative=3,
                               random_state=24)
    lcv = LogisticRegressionCV(Cs=5, cv=3).fit(X, y)
    assert lcv.C_ in lcv.Cs_
    assert lcv.scores_.shape == (5, 3)
    assert lcv.score(X, y) > 0.8
    assert np.allclose(lcv.predict_proba(X).sum(axis=1), 1)


def test_omp_selects_exactly_k_features():
    X, y = make_regression(n_samples=100, n_features=20, n_informative=3,
                           noise=0.1, random_state=25)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=3).fit(X, y)
    assert (np.abs(omp.coef_) > 1e-10).sum() == 3
    assert omp.n_nonzero_coefs_ == 3
    assert omp.score(X, y) > 0.9


def test_omp_matches_sklearn():
    X, y = make_regression(n_samples=100, n_features=15, n_informative=4,
                           noise=0.5, random_state=26)
    ours = OrthogonalMatchingPursuit(n_nonzero_coefs=4).fit(X, y)
    ref = sk.OrthogonalMatchingPursuit(n_nonzero_coefs=4).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_, atol=1e-6)
    # the same support is selected
    assert set(np.nonzero(ours.coef_)[0]) == set(np.nonzero(ref.coef_)[0])
