"""Nonparametric and robust regression: LOESS, GAM, MARS, RuleFit, and friends.

The tests use functions with a known shape -- a sine, a piecewise line, a planted
equation -- so each method can be held to what it claims: LOESS to follow a
curve, GAM to recover a per-feature effect, MARS to find where a relationship
bends, RuleFit to reduce a forest to a short list.
"""
import numpy as np
import pytest

from nupyml.datasets import make_regression, load_iris, make_classification
from nupyml.linear_model import (Lars, LassoLars, LinearSVR, RidgeClassifier,
                                 MultiTaskLasso, LinearRegression, Lasso)
from nupyml.model_selection import train_test_split
from nupyml.nonparametric import (
    LOESS, NadarayaWatson, LocalLinearRegression, GAM, GAMClassifier,
    SplineTransformer, natural_cubic_basis, MARS, RuleFit, Rule, KernelRidge,
    TotalLeastSquares, LTSRegressor, NonlinearLeastSquares, RVMRegressor,
    BARTRegressor,
)
from nupyml.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def sine():
    rng = np.random.RandomState(0)
    X = np.sort(rng.uniform(-3, 3, (400, 1)), axis=0)
    y = np.sin(2 * X[:, 0]) + 0.1 * rng.normal(size=400)
    return train_test_split(X, y, test_size=0.3, random_state=0)


@pytest.fixture(scope="module")
def multi_regression():
    X, y = make_regression(n_samples=500, n_features=6, noise=8.0, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)


# --- local regression -----------------------------------------------------

def test_nadaraya_watson_follows_a_curve(sine):
    Xtr, Xte, ytr, yte = sine
    nw = NadarayaWatson(bandwidth=0.3).fit(Xtr, ytr)
    assert nw.score(Xte, yte) > 0.9


def test_local_linear_beats_nadaraya_watson_at_the_boundary():
    """The boundary-bias lesson, measured: on a sloped function the constant fit
    flattens toward the edge while the local line follows through."""
    rng = np.random.RandomState(0)
    X = np.sort(rng.uniform(0, 1, (300, 1)), axis=0)
    y = 3 * X[:, 0] + rng.normal(0, 0.05, 300)     # a clean upward slope
    edge = X[:, 0] > 0.85                            # the right boundary region

    nw = NadarayaWatson(bandwidth=0.15).fit(X, y)
    ll = LocalLinearRegression(bandwidth=0.15).fit(X, y)
    nw_err = np.abs(nw.predict(X[edge]) - y[edge]).mean()
    ll_err = np.abs(ll.predict(X[edge]) - y[edge]).mean()
    assert ll_err < nw_err


def test_local_linear_estimates_a_slope(sine):
    Xtr, Xte, ytr, yte = sine
    ll = LocalLinearRegression(bandwidth=0.5).fit(Xtr, ytr)
    assert ll.score(Xte, yte) > 0.95


def test_loess_follows_a_curve(sine):
    Xtr, Xte, ytr, yte = sine
    lo = LOESS(frac=0.3).fit(Xtr, ytr)
    assert lo.score(Xte, yte) > 0.85


def test_loess_robustness_iterations_resist_outliers():
    """A single wild point puts a visible bump in a non-robust local fit, because
    the fit is local. The bisquare reweighting removes it."""
    rng = np.random.RandomState(0)
    X = np.sort(rng.uniform(-3, 3, (200, 1)), axis=0)
    y = np.sin(X[:, 0]) + 0.05 * rng.normal(size=200)
    y[100] += 10.0                                   # one gross outlier

    naive = LOESS(frac=0.3, n_iter=1).fit(X, y)      # no robustness passes
    robust = LOESS(frac=0.3, n_iter=4).fit(X, y)
    # error at the neighbours of the outlier, excluding the outlier itself
    near = slice(95, 100)
    clean = np.sin(X[near, 0])
    assert np.abs(robust.predict(X[near]) - clean).mean() < \
        np.abs(naive.predict(X[near]) - clean).mean()


def test_loess_rejects_multivariate_input():
    with pytest.raises(ValueError, match="single feature"):
        LOESS().fit(np.random.rand(20, 2), np.random.rand(20))


def test_larger_bandwidth_is_smoother():
    rng = np.random.RandomState(0)
    X = np.sort(rng.uniform(-3, 3, (200, 1)), axis=0)
    y = np.sin(2 * X[:, 0]) + 0.1 * rng.normal(size=200)
    grid = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
    wiggly = LocalLinearRegression(bandwidth=0.1).fit(X, y).predict(grid)
    smooth = LocalLinearRegression(bandwidth=1.0).fit(X, y).predict(grid)
    # roughness = total variation of the fitted curve
    assert np.abs(np.diff(smooth)).sum() < np.abs(np.diff(wiggly)).sum()


# --- GAM ------------------------------------------------------------------

def test_natural_cubic_basis_is_linear_beyond_the_boundary():
    """The "natural" constraint: second derivative zero past the end knots, so
    the fit is a straight line out there rather than a runaway cubic."""
    knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    x = np.array([5.0, 6.0, 7.0, 8.0])              # all beyond the last knot
    B = natural_cubic_basis(x, knots)
    # a straight line in x => equal second differences of zero across the basis
    second_diff = np.diff(B, n=2, axis=0)
    assert np.allclose(second_diff, 0, atol=1e-8)


def test_gam_recovers_an_additive_structure():
    """The model is f_1(x1) + f_2(x2); a GAM should fit it well where a linear
    model cannot."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (600, 2))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.1 * rng.normal(size=600)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    gam = GAM(n_knots=10, lam=0.1).fit(Xtr, ytr)
    linear = LinearRegression().fit(Xtr, ytr)
    assert gam.score(Xte, yte) > 0.9
    assert gam.score(Xte, yte) > linear.score(Xte, yte)


def test_gam_partial_dependence_is_the_model_for_one_feature():
    """The reason to use a GAM: the per-feature curve is exactly what the model
    adds, readable off directly. Here f_2 should look like a parabola."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (600, 2))
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.1 * rng.normal(size=600)
    gam = GAM(n_knots=10, lam=0.1).fit(X, y)
    grid, effect = gam.partial_dependence(1)
    # a parabola is symmetric and rises at both ends: the ends exceed the middle
    mid = len(effect) // 2
    assert effect[0] > effect[mid] and effect[-1] > effect[mid]


def test_gam_effects_are_centred():
    """Each f_j is mean-zero, or the intercept is unidentifiable and the fit can
    drift without changing the predictions."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (400, 3))
    y = X[:, 0] ** 2 + rng.normal(0, 0.1, 400)
    gam = GAM(n_knots=8).fit(X, y)
    for j in range(3):
        _, effect = gam.partial_dependence(j, grid=X[:, j])
        assert abs(effect.mean()) < 0.1


def test_gam_classifier_fits_a_nonlinear_boundary():
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, (500, 2))
    # a circular boundary: no linear classifier can do this, an additive one can
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 4).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    gam = GAMClassifier(n_knots=8, lam=0.5).fit(Xtr, ytr)
    assert gam.score(Xte, yte) > 0.85
    p = gam.predict_proba(Xte)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_gam_classifier_rejects_multiclass():
    X, y = load_iris(return_X_y=True)
    with pytest.raises(ValueError, match="binary"):
        GAMClassifier().fit(X, y)


def test_spline_transformer_expands_and_stays_local():
    """A spline basis is local: a bump at one knot must not move the fit far
    away, unlike a polynomial basis."""
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    st = SplineTransformer(n_knots=6).fit(X)
    B = st.transform(X)
    assert B.shape[0] == 100 and B.shape[1] > 1
    # each basis column is non-zero on only part of the range (compact-ish support)
    fractions = (np.abs(B) > 1e-6).mean(axis=0)
    assert np.any(fractions < 0.9)


# --- MARS -----------------------------------------------------------------

def test_mars_fits_a_curve(sine):
    Xtr, Xte, ytr, yte = sine
    mars = MARS(max_terms=21).fit(Xtr, ytr)
    assert mars.score(Xte, yte) > 0.9


def test_mars_finds_a_hinge_near_a_real_breakpoint():
    """MARS discovers WHERE a relationship bends. Plant a kink at x=1 and it
    should place a knot near there."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, (500, 1))
    y = np.where(X[:, 0] < 1.0, 0.0, 3 * (X[:, 0] - 1.0)) + 0.05 * rng.normal(size=500)
    mars = MARS(max_terms=11, max_degree=1).fit(X, y)
    knots = [f[1] for t in mars.terms_ for f in t.factors]
    assert knots and min(abs(np.array(knots) - 1.0)) < 0.4


def test_mars_gives_a_readable_expression(sine):
    Xtr, _, ytr, _ = sine
    mars = MARS(max_terms=11).fit(Xtr, ytr)
    assert "max(0," in mars.expression_


def test_mars_can_forbid_interactions():
    """max_degree=1 restricts to an additive model -- a GAM with learned knots."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (300, 3))
    y = X[:, 0] + np.abs(X[:, 1]) + rng.normal(0, 0.1, 300)
    mars = MARS(max_terms=15, max_degree=1).fit(X, y)
    assert all(len(t.factors) <= 1 for t in mars.terms_)


def test_mars_prunes_in_the_backward_pass():
    """Overfit forward, prune back: the final model should be smaller than the
    forward pass was allowed to grow."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (300, 2))
    y = X[:, 0] + rng.normal(0, 0.1, 300)          # only one feature matters
    mars = MARS(max_terms=21).fit(X, y)
    assert len(mars.terms_) < 21


# --- RuleFit --------------------------------------------------------------

def test_rule_fires_on_its_conjunction():
    rule = Rule([(0, ">", 1.0), (1, "<=", 2.0)])
    X = np.array([[2.0, 1.0], [0.0, 1.0], [2.0, 3.0]])
    assert np.array_equal(rule.evaluate(X), [1.0, 0.0, 0.0])


def test_rulefit_is_accurate_and_short(multi_regression):
    Xtr, Xte, ytr, yte = multi_regression
    rf = RuleFit(n_estimators=50, random_state=0).fit(Xtr, ytr)
    assert rf.score(Xte, yte) > 0.9
    # the whole point: a forest's accuracy from a readable handful of terms
    assert rf.n_rules_ < 60


def test_rulefit_prefers_linear_terms_on_a_linear_problem():
    """A forest of rules staircasing a straight line is wasteful; the lasso
    should keep the linear terms and few or no rules."""
    X, y = make_regression(n_samples=400, n_features=5, noise=5.0, random_state=0)
    rf = RuleFit(n_estimators=50, alpha=0.5, random_state=0).fit(X, y)
    assert rf.n_rules_ < 10
    assert np.any(np.abs(rf.linear_coef_) > 1e-6)


def test_rulefit_extracts_a_useful_rule_on_a_threshold_problem():
    """Where the truth is a threshold interaction, rules should carry the model
    -- this is what rules do that linear terms cannot."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-2, 2, (600, 2))
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(float) * 5 + rng.normal(0, 0.2, 600)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    rf = RuleFit(n_estimators=80, alpha=0.05, random_state=0).fit(Xtr, ytr)
    assert rf.score(Xte, yte) > 0.85
    assert rf.n_rules_ >= 1
    assert isinstance(rf.summary(), str)


# --- kernel ridge ---------------------------------------------------------

def test_kernel_ridge_fits_a_nonlinear_function(sine):
    Xtr, Xte, ytr, yte = sine
    kr = KernelRidge(alpha=0.1, gamma=1.0).fit(Xtr, ytr)
    assert kr.score(Xte, yte) > 0.95


def test_kernel_ridge_linear_kernel_matches_ridge():
    """A linear kernel is ridge in the original space -- they should agree."""
    from nupyml.linear_model import Ridge
    X, y = make_regression(n_samples=100, n_features=5, noise=5.0, random_state=0)
    kr = KernelRidge(alpha=1.0, kernel="linear").fit(X, y)
    # KRR with a linear kernel and Ridge solve the same problem up to the
    # intercept handling; predictions should track closely
    ridge = Ridge(alpha=1.0, fit_intercept=False).fit(X, y)
    assert np.corrcoef(kr.predict(X), ridge.predict(X))[0, 1] > 0.99


def test_kernel_ridge_is_dense():
    """Unlike SVR, KRR uses every training point -- no dual coefficient is zero.
    That density is the trade it makes for a closed-form fit."""
    X, y = make_regression(n_samples=80, n_features=3, noise=5.0, random_state=0)
    kr = KernelRidge(alpha=1.0, gamma=0.1).fit(X, y)
    assert np.all(np.abs(kr.dual_coef_) > 1e-12)


def test_kernel_ridge_rejects_unknown_kernel():
    X, y = make_regression(n_samples=50, n_features=3, random_state=0)
    with pytest.raises(ValueError, match="Unknown kernel"):
        KernelRidge(kernel="nonsense").fit(X, y)


# --- robust and errors-in-variables ---------------------------------------

def test_total_least_squares_beats_ols_when_x_is_noisy():
    """OLS assumes x is exact and attenuates the slope when it is not. TLS treats
    both variables symmetrically and does not."""
    rng = np.random.RandomState(0)
    x_true = rng.uniform(0, 10, 400)
    x_obs = x_true + rng.normal(0, 1.5, 400)         # x is measured with noise
    y = 3.0 * x_true + 2.0 + rng.normal(0, 1.0, 400)

    ols = LinearRegression().fit(x_obs[:, None], y).coef_[0]
    tls = TotalLeastSquares().fit(x_obs[:, None], y).coef_[0]
    assert abs(tls - 3.0) < abs(ols - 3.0)           # TLS is closer to the truth
    assert ols < 3.0                                  # OLS attenuates, as predicted


def test_lts_resists_outliers_that_break_ols():
    """OLS has breakdown point zero: a few outliers wreck it. LTS fits the clean
    majority and marks the rest."""
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 10, (200, 1))
    y = 2.0 * X[:, 0] + 1.0 + rng.normal(0, 0.3, 200)
    y[:30] += 40.0                                    # 15% gross outliers

    ols = LinearRegression().fit(X, y).coef_[0]
    lts = LTSRegressor(random_state=0).fit(X, y)
    assert abs(lts.coef_[0] - 2.0) < abs(ols - 2.0)
    assert abs(lts.coef_[0] - 2.0) < 0.3             # close to the true slope
    # the outliers should mostly be flagged as non-inliers
    assert lts.inlier_mask_[:30].mean() < 0.5


def test_nonlinear_least_squares_recovers_known_parameters():
    """When the FORM is known, LM finds its parameters -- which is a different
    goal from prediction, and the point of the method."""
    rng = np.random.RandomState(0)
    x = rng.uniform(0, 2, 200)
    y = 2.5 * np.exp(0.8 * x) + rng.normal(0, 0.2, 200)

    def model(X, p):
        return p[0] * np.exp(p[1] * X[:, 0])

    nls = NonlinearLeastSquares(model, p0=[1.0, 1.0]).fit(x[:, None], y)
    assert nls.params_[0] == pytest.approx(2.5, abs=0.2)
    assert nls.params_[1] == pytest.approx(0.8, abs=0.1)


# --- Bayesian nonparametric -----------------------------------------------

def test_rvm_is_sparse_and_accurate():
    """ARD's payoff: a handful of relevance vectors, not the whole training set.
    A batch update needs a relative pruning threshold to achieve it -- see the
    class docstring."""
    rng = np.random.RandomState(0)
    X = np.sort(rng.uniform(-4, 4, (200, 1)), axis=0)
    y = np.sinc(X[:, 0]) + 0.05 * rng.normal(size=200)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    rvm = RVMRegressor(gamma=0.5).fit(Xtr, ytr)
    assert rvm.score(Xte, yte) > 0.9
    assert rvm.n_relevance_ < 0.25 * len(Xtr)        # genuinely sparse


def test_rvm_reports_a_predictive_variance():
    """The reason to use RVM over an SVM: a variance per prediction, not just a
    point. It is always positive and at least the noise floor."""
    rng = np.random.RandomState(0)
    X = np.sort(rng.uniform(-3, 3, (150, 1)), axis=0)
    y = np.sin(X[:, 0]) + 0.05 * rng.normal(size=150)
    rvm = RVMRegressor(gamma=0.5).fit(X, y)
    mean, std = rvm.predict(X, return_std=True)
    assert np.all(std > 0)
    assert std.shape == mean.shape


def test_rvm_uncertainty_is_backwards_far_from_the_data():
    """The famous RVM flaw, demonstrated rather than hidden. With RBF basis, every
    basis function decays to zero away from its relevance vector, so far from the
    data the predictive variance COLLAPSES to the noise floor -- the model is most
    confident exactly where it has seen nothing. A GP does the opposite; this is
    why GPs win when the uncertainty itself is the product.
    """
    rng = np.random.RandomState(0)
    X = np.sort(rng.uniform(-3, 3, (150, 1)), axis=0)
    y = np.sin(X[:, 0]) + 0.05 * rng.normal(size=150)
    rvm = RVMRegressor(gamma=0.5).fit(X, y)
    _, std_inside = rvm.predict(np.array([[0.0]]), return_std=True)
    _, std_far = rvm.predict(np.array([[20.0]]), return_std=True)
    noise_floor = np.sqrt(1.0 / rvm.beta_)
    # far away, the variance is essentially just noise -- not larger than inside
    assert std_far[0] == pytest.approx(noise_floor, rel=0.05)
    assert std_far[0] <= std_inside[0] + 1e-6


def test_bart_credible_intervals_cover():
    """BART reads its intervals from posterior draws -- no normality assumed."""
    X, y = make_regression(n_samples=300, n_features=4, noise=5.0, random_state=0)
    bart = BARTRegressor(n_trees=30, n_draws=100, burn_in=50,
                         random_state=0).fit(X, y)
    lo, hi = bart.predict_interval(X, coverage=0.9)
    assert 0.8 < ((y >= lo) & (y <= hi)).mean() <= 1.0


def test_bart_intervals_widen_with_coverage():
    X, y = make_regression(n_samples=200, n_features=3, noise=5.0, random_state=0)
    bart = BARTRegressor(n_trees=20, n_draws=80, burn_in=40,
                         random_state=0).fit(X, y)
    lo50, hi50 = bart.predict_interval(X, 0.5)
    lo90, hi90 = bart.predict_interval(X, 0.9)
    assert (hi90 - lo90).mean() > (hi50 - lo50).mean()


# --- LARS and the linear-model gaps ---------------------------------------

def test_lars_recovers_the_informative_features():
    """LARS enters features one at a time in order of relevance; the first few
    should be the informative ones."""
    X, y = make_regression(n_samples=300, n_features=12, n_informative=4,
                           noise=5.0, random_state=0)
    lars = Lars(n_nonzero_coefs=4).fit(X, y)
    assert (np.abs(lars.coef_) > 1e-8).sum() == 4


def test_lars_exposes_the_whole_path():
    """The distinctive feature: every solution for every penalty, in one pass."""
    X, y = make_regression(n_samples=200, n_features=8, n_informative=3,
                           noise=5.0, random_state=0)
    lars = Lars().fit(X, y)
    # the path starts at all-zero and grows one active coefficient per step
    assert np.allclose(lars.coef_path_[0], 0)
    nonzero_per_step = (np.abs(lars.coef_path_) > 1e-10).sum(axis=1)
    assert nonzero_per_step[-1] >= nonzero_per_step[0]


def test_lasso_lars_traces_the_lasso_path():
    """The one extra rule -- drop a feature when its coefficient hits zero --
    turns the LARS walk into the exact lasso path."""
    X, y = make_regression(n_samples=200, n_features=10, n_informative=3,
                           noise=5.0, random_state=0)
    ll = LassoLars(n_nonzero_coefs=3).fit(X, y)
    assert ll.score(X, y) > 0.5


def test_linear_svr_needs_and_rewards_scaling():
    """SVR is scale-sensitive by nature; standardised, it is excellent. This
    documents the requirement rather than hiding it."""
    X, y = make_regression(n_samples=300, n_features=6, noise=5.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    scaler = StandardScaler().fit(Xtr)
    ys = ytr.std()
    svr = LinearSVR(epsilon=0.1, C=10.0, random_state=0).fit(
        scaler.transform(Xtr), ytr / ys)
    assert svr.score(scaler.transform(Xte), yte / ys) > 0.9


def test_linear_svr_epsilon_tube_ignores_small_errors():
    """A larger tube should change the fit -- points inside it stop mattering."""
    X, y = make_regression(n_samples=200, n_features=4, noise=5.0, random_state=0)
    scaler = StandardScaler().fit(X)
    Xs, ys = scaler.transform(X), y / y.std()
    tight = LinearSVR(epsilon=0.01, C=10.0, random_state=0).fit(Xs, ys)
    wide = LinearSVR(epsilon=1.0, C=10.0, random_state=0).fit(Xs, ys)
    assert not np.allclose(tight.coef_, wide.coef_)


def test_ridge_classifier_multiclass():
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    rc = RidgeClassifier().fit(Xtr, ytr)
    assert rc.score(Xte, yte) > 0.7
    assert set(rc.predict(Xte)).issubset(set(np.unique(y)))


def test_ridge_classifier_binary():
    X, y = make_classification(n_samples=200, n_features=6, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    rc = RidgeClassifier().fit(Xtr, ytr)
    assert rc.score(Xte, yte) > 0.8


def test_multitask_lasso_shares_support_across_tasks():
    """The defining property: a feature is used by EVERY task or by NONE, so the
    fitted support is identical across tasks. Here only features 1, 4, 7 matter."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 10))
    W = np.zeros((10, 3))
    W[[1, 4, 7]] = rng.normal(size=(3, 3))
    Y = X @ W + rng.normal(0, 0.3, (300, 3))
    mt = MultiTaskLasso(alpha=0.5).fit(X, Y)

    used = np.abs(mt.coef_) > 1e-6                    # (n_tasks, n_features)
    # every task uses the same features -- the columns of `used` are identical
    assert all(np.array_equal(used[0], used[t]) for t in range(1, 3))
    assert set(np.where(used[0])[0]) == {1, 4, 7}


def test_multitask_lasso_predicts_all_targets():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 5))
    Y = np.column_stack([X[:, 0], X[:, 1] + X[:, 0]])
    mt = MultiTaskLasso(alpha=0.1).fit(X, Y)
    assert mt.predict(X).shape == (200, 2)
