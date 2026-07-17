"""Survival extensions, change-point detection, and copulas.

Survival tests plant covariate-driven risk and check the models rank it
(C-index > 0.5); change-point tests plant mean shifts at known indices and check
recovery; copula tests fit dependence and check the sampled data reproduces it.
"""
import numpy as np
import pytest

from nupyml.survival import (concordance_index, RandomSurvivalForest, AFT, CoxPH,
                            KaplanMeier)
from nupyml.changepoint import pelt, binary_segmentation, cusum, bocpd
from nupyml.copula import (GaussianCopula, StudentTCopula, ClaytonCopula,
                          GumbelCopula, FrankCopula)


@pytest.fixture
def survival_data():
    rng = np.random.RandomState(0)
    n = 500
    X = rng.normal(size=(n, 3))
    risk = np.exp(X @ [0.8, -0.5, 0.3])
    t = rng.exponential(1 / risk)
    c = rng.exponential(2, n)
    dur = np.minimum(t, c)
    ev = (t <= c).astype(int)
    return X[:350], X[350:], dur[:350], dur[350:], ev[:350], ev[350:]


# --- survival -------------------------------------------------------------

def test_concordance_index_of_perfect_ranking_is_one():
    """A risk score that perfectly orders the failure times scores C = 1."""
    dur = np.array([1.0, 2.0, 3.0, 4.0])
    ev = np.array([1, 1, 1, 1])
    risk = np.array([4.0, 3.0, 2.0, 1.0])       # higher risk fails sooner
    assert concordance_index(dur, ev, risk) == pytest.approx(1.0)


def test_concordance_index_of_reversed_ranking_is_zero():
    dur = np.array([1.0, 2.0, 3.0, 4.0])
    ev = np.array([1, 1, 1, 1])
    risk = np.array([1.0, 2.0, 3.0, 4.0])       # exactly backwards
    assert concordance_index(dur, ev, risk) == pytest.approx(0.0)


def test_concordance_index_respects_censoring():
    """A subject censored early cannot be ordered against a later one, so it does
    not count as a comparable pair."""
    dur = np.array([2.0, 5.0])
    ev = np.array([0, 1])                        # first is censored at 2
    # the pair is not comparable (we do not know the censored subject's true time
    # relative to the other), so with no comparable pairs C defaults to 0.5
    assert concordance_index(dur, ev, np.array([1.0, 2.0])) == 0.5


def test_random_survival_forest_ranks_risk(survival_data):
    Xtr, Xte, dtr, dte, etr, ete = survival_data
    rsf = RandomSurvivalForest(n_estimators=30, random_state=0).fit(Xtr, dtr, etr)
    assert concordance_index(dte, ete, rsf.predict_risk(Xte)) > 0.6


def test_aft_ranks_risk_and_predicts_positive_times(survival_data):
    Xtr, Xte, dtr, dte, etr, ete = survival_data
    aft = AFT().fit(Xtr, dtr, etr)
    assert concordance_index(dte, ete, aft.predict_risk(Xte)) > 0.6
    # median survival times are genuine (positive) times
    assert np.all(aft.predict_median_survival(Xte) > 0)


def test_survival_models_agree_on_ranking(survival_data):
    """Cox, RSF and AFT should all rank risk well above chance on the same data."""
    Xtr, Xte, dtr, dte, etr, ete = survival_data
    cox = CoxPH().fit(Xtr, dtr, etr)
    for model in (cox, RandomSurvivalForest(n_estimators=30, random_state=0).fit(
            Xtr, dtr, etr), AFT().fit(Xtr, dtr, etr)):
        assert concordance_index(dte, ete, model.predict_risk(Xte)) > 0.6


# --- change-point ---------------------------------------------------------

@pytest.fixture
def two_shifts():
    """Mean shifts 0 -> 5 -> 0 at indices 100 and 200."""
    rng = np.random.RandomState(0)
    return np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 100),
                           rng.normal(0, 1, 100)])


def test_pelt_finds_both_change_points(two_shifts):
    cps = pelt(two_shifts)
    assert len(cps) == 2
    assert all(min(abs(cp - true) for true in (100, 200)) <= 5 for cp in cps)


def test_binary_segmentation_finds_both_change_points(two_shifts):
    cps = binary_segmentation(two_shifts)
    assert len([c for c in cps if abs(c - 100) <= 5 or abs(c - 200) <= 5]) == 2


def test_pelt_penalty_controls_the_number_of_change_points(two_shifts):
    """More penalty -> fewer change-points. The knob that stops over-segmentation."""
    few = pelt(two_shifts, penalty=1000.0)
    many = pelt(two_shifts, penalty=5.0)
    assert len(few) <= len(many)


def test_pelt_finds_nothing_in_a_stationary_signal():
    stationary = np.random.RandomState(0).normal(0, 1, 300)
    assert len(pelt(stationary)) == 0


def test_cusum_locates_a_single_shift():
    rng = np.random.RandomState(0)
    signal = np.concatenate([rng.normal(0, 1, 150), rng.normal(4, 1, 150)])
    cp = cusum(signal)
    assert cp is not None and abs(cp - 150) < 15


def test_cusum_returns_none_when_stable():
    stable = np.random.RandomState(0).normal(0, 1, 300)
    assert cusum(stable) is None


def test_bocpd_run_length_collapses_at_a_change():
    """The run length grows steadily, then drops toward zero when a change
    occurs -- the posterior resetting to 'the last change was just now'."""
    rng = np.random.RandomState(0)
    signal = np.concatenate([rng.normal(0, 1, 150), rng.normal(6, 1, 150)])
    rl = bocpd(signal)
    assert rl[148] > 50                          # run has grown before the change
    assert rl[152] < 10                          # and reset just after it


# --- copulas --------------------------------------------------------------

@pytest.fixture
def correlated_data():
    rng = np.random.RandomState(0)
    return rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], 800)


def test_gaussian_copula_reproduces_correlation(correlated_data):
    gc = GaussianCopula().fit(correlated_data)
    sample = gc.sample(2000, random_state=1)
    assert np.corrcoef(sample.T)[0, 1] == pytest.approx(
        gc.correlation_[0, 1], abs=0.1)


def test_gaussian_copula_samples_are_uniform(correlated_data):
    """A copula's output marginals are uniform on [0,1] by construction."""
    gc = GaussianCopula().fit(correlated_data)
    sample = gc.sample(2000, random_state=1)
    assert sample.min() >= 0 and sample.max() <= 1
    assert abs(sample[:, 0].mean() - 0.5) < 0.05


def test_student_t_copula_has_heavier_tail_dependence(correlated_data):
    """The t copula should produce more JOINT extremes than the Gaussian at the
    same correlation -- its defining tail dependence."""
    gc = GaussianCopula().fit(correlated_data)
    tc = StudentTCopula(df=2).fit(correlated_data)
    g = gc.sample(5000, random_state=1)
    t = tc.sample(5000, random_state=1)
    # count joint lower-tail events (both variables below the 5th percentile)
    g_joint = np.mean((g[:, 0] < 0.05) & (g[:, 1] < 0.05))
    t_joint = np.mean((t[:, 0] < 0.05) & (t[:, 1] < 0.05))
    assert t_joint > g_joint


@pytest.mark.parametrize("copula_cls", [ClaytonCopula, GumbelCopula, FrankCopula])
def test_archimedean_copula_captures_positive_dependence(copula_cls,
                                                         correlated_data):
    cop = copula_cls().fit(correlated_data)
    sample = cop.sample(2000, random_state=1)
    assert np.corrcoef(sample.T)[0, 1] > 0.3     # positive dependence preserved
    assert sample.min() >= 0 and sample.max() <= 1


def test_clayton_has_lower_tail_dependence():
    """Clayton should show MORE joint LOW extremes than joint HIGH -- its
    asymmetric lower-tail signature."""
    rng = np.random.RandomState(0)
    data = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 800)
    clayton = ClaytonCopula().fit(data)
    s = clayton.sample(8000, random_state=1)
    lower = np.mean((s[:, 0] < 0.1) & (s[:, 1] < 0.1))
    upper = np.mean((s[:, 0] > 0.9) & (s[:, 1] > 0.9))
    assert lower > upper
