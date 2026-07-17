"""F9: additive hazards, competing risks, parametric AFT, time-dependent metrics.

Held to their defining properties: additive hazards recovers a time-varying
covariate effect; the Aalen-Johansen CIFs are non-decreasing and sum to 1-S(t);
the Weibull AFT discriminates and its coefficient has the right sign;
time-dependent AUC and the IPCW Brier score reward a good risk score / prediction.
"""
import numpy as np
import pytest

from nupyml.survival import (AalenAdditiveHazards, AalenJohansen, WeibullAFT,
                            brier_score, integrated_brier_score,
                            time_dependent_auc, concordance_index)


def _survival_data(seed=0, n=400):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    risk = X[:, 0] * 1.5                             # feature 0 drives risk
    base = rng.exponential(scale=np.exp(-risk))     # higher risk -> shorter life
    cens = rng.exponential(scale=2.0, size=n)
    t = np.minimum(base, cens)
    e = (base <= cens).astype(int)
    return X, t, e


# --- Weibull AFT ----------------------------------------------------------

def test_weibull_aft_discriminates_and_has_right_sign():
    X, t, e = _survival_data()
    aft = WeibullAFT().fit(X, t, e)
    # feature 0 shortens life -> negative log-time coefficient (index 1 after intercept)
    assert aft.coef_[1] < 0
    # -median as a risk score should be concordant with the outcomes
    ci = concordance_index(t, e, -aft.predict_median(X))
    assert ci > 0.7


def test_weibull_aft_survival_is_monotone_in_time():
    X, t, e = _survival_data()
    aft = WeibullAFT().fit(X, t, e)
    s_early = aft.predict_survival(X, 0.2)
    s_late = aft.predict_survival(X, 2.0)
    assert np.all(s_late <= s_early + 1e-9)         # survival never increases
    assert np.all((s_early >= 0) & (s_early <= 1))


# --- Aalen additive -------------------------------------------------------

def test_aalen_additive_recovers_covariate_effect():
    X, t, e = _survival_data()
    aa = AalenAdditiveHazards().fit(X, t, e)
    ch = aa.predict_cumulative_hazard(X)
    # cumulative hazard should track the risk feature
    assert abs(np.corrcoef(X[:, 0], ch)[0, 1]) > 0.5


def test_aalen_cumulative_hazard_grows_over_time():
    X, t, e = _survival_data()
    aa = AalenAdditiveHazards().fit(X, t, e)
    early = aa.predict_cumulative_hazard(X, t=np.percentile(t, 20)).mean()
    late = aa.predict_cumulative_hazard(X, t=np.percentile(t, 90)).mean()
    assert late >= early                            # cumulative hazard accumulates


# --- competing risks ------------------------------------------------------

def test_aalen_johansen_cifs_sum_to_one_minus_survival():
    rng = np.random.RandomState(1)
    n = 500
    t = rng.exponential(1.0, n)
    ev = rng.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3])   # censored / cause1 / cause2
    aj = AalenJohansen().fit(t, ev)
    tmax = t.max()
    total_cif = aj.predict_cif(1, tmax) + aj.predict_cif(2, tmax)
    # overall event probability by tmax = 1 - S(tmax); CIFs must partition it
    from nupyml.survival import KaplanMeier
    km = KaplanMeier().fit(t, (ev != 0).astype(int))
    one_minus_S = 1 - km.predict(tmax)
    assert abs(total_cif - one_minus_S) < 0.05


def test_aalen_johansen_cif_is_non_decreasing():
    rng = np.random.RandomState(2)
    n = 400
    t = rng.exponential(1.0, n)
    ev = rng.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3])
    aj = AalenJohansen().fit(t, ev)
    inc = aj.cif_[1]["incidence"]
    assert np.all(np.diff(inc) >= -1e-9)
    assert inc[-1] <= 1.0


# --- time-dependent metrics -----------------------------------------------

def test_time_dependent_auc_rewards_informative_risk():
    X, t, e = _survival_data()
    aft = WeibullAFT().fit(X, t, e)
    horizon = np.percentile(t, 60)
    auc_good = time_dependent_auc(t, e, -aft.predict_median(X), horizon)
    rng = np.random.RandomState(0)
    auc_rand = time_dependent_auc(t, e, rng.randn(len(t)), horizon)
    assert auc_good > 0.7
    assert abs(auc_rand - 0.5) < 0.12


def test_brier_score_lower_for_good_predictions():
    X, t, e = _survival_data()
    aft = WeibullAFT().fit(X, t, e)
    horizon = np.percentile(t, 50)
    good = brier_score(t, e, aft.predict_survival(X, horizon), horizon)
    naive = brier_score(t, e, np.full(len(t), 0.5), horizon)
    assert good < naive
    assert 0 <= good <= 1


def test_integrated_brier_score_is_a_valid_average():
    X, t, e = _survival_data()
    aft = WeibullAFT().fit(X, t, e)
    grid = np.percentile(t, [20, 40, 60, 80])
    ibs = integrated_brier_score(t, e, lambda tt: aft.predict_survival(X, tt), grid)
    assert 0 <= ibs <= 1
