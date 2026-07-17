"""Fairness metrics and mitigation.

A biased dataset is built where the sensitive attribute is correlated with the
label AND leaks into the features, so an ordinary classifier discriminates. Each
metric must DETECT that, and each mitigator must REDUCE the disparity relative to
the unmitigated model -- the property each one advertises.
"""
import numpy as np
import pytest

from nupyml.linear_model import LogisticRegression
from nupyml.fairness import (selection_rate, demographic_parity_difference,
                            disparate_impact_ratio, equal_opportunity_difference,
                            equalized_odds_difference,
                            predictive_parity_difference, Reweighing,
                            CorrelationRemover, ThresholdOptimizer,
                            ExponentiatedGradientReduction)


def _biased_data(seed=0, n=1200):
    """Sensitive attribute s in {0,1}; group 1 has a higher base rate AND a
    feature shifted by s, so a classifier learns to over-select group 1."""
    rng = np.random.RandomState(seed)
    s = rng.randint(0, 2, n)
    x0 = rng.randn(n) + 1.5 * s            # a feature that leaks the group
    x1 = rng.randn(n)
    # label depends on x1 and on s (biased base rates)
    logit = 1.2 * x1 + 1.5 * s - 0.5
    y = (rng.rand(n) < 1 / (1 + np.exp(-logit))).astype(int)
    X = np.column_stack([x0, x1])
    return X, y, s


def _split(X, y, s, frac=0.6):
    n = len(y)
    rng = np.random.RandomState(1)
    idx = rng.permutation(n)
    k = int(frac * n)
    tr, te = idx[:k], idx[k:]
    return X[tr], y[tr], s[tr], X[te], y[te], s[te]


# --- metrics --------------------------------------------------------------

def test_metrics_detect_disparity_on_a_biased_model():
    X, y, s = _biased_data()
    clf = LogisticRegression(max_iter=500).fit(X, y)
    pred = clf.predict(X)
    assert demographic_parity_difference(pred, s) > 0.1
    assert disparate_impact_ratio(pred, s) < 0.9        # below parity (1.0)
    assert equalized_odds_difference(y, pred, s) >= 0.0


def test_selection_rate_matches_manual():
    y_pred = np.array([1, 1, 0, 0, 1, 0])
    s = np.array([0, 0, 0, 1, 1, 1])
    rates = selection_rate(y_pred, s)
    assert rates[0] == pytest.approx(2 / 3)
    assert rates[1] == pytest.approx(1 / 3)


def test_perfect_parity_gives_zero_difference():
    y_pred = np.array([1, 0, 1, 0])
    s = np.array([0, 0, 1, 1])
    assert demographic_parity_difference(y_pred, s) == 0.0
    assert disparate_impact_ratio(y_pred, s) == pytest.approx(1.0)


def test_predictive_and_equal_opportunity_are_defined():
    X, y, s = _biased_data()
    pred = LogisticRegression(max_iter=500).fit(X, y).predict(X)
    assert 0.0 <= equal_opportunity_difference(y, pred, s) <= 1.0
    assert 0.0 <= predictive_parity_difference(y, pred, s) <= 1.0


# --- pre-processing -------------------------------------------------------

def test_reweighing_reduces_demographic_parity_gap():
    Xtr, ytr, str_, Xte, yte, ste = _split(*_biased_data())
    base = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    w = Reweighing().fit_transform(Xtr, ytr, str_)
    assert np.all(w > 0)
    fair = LogisticRegression(max_iter=500).fit(Xtr, ytr, sample_weight=w)
    gap_base = demographic_parity_difference(base.predict(Xte), ste)
    gap_fair = demographic_parity_difference(fair.predict(Xte), ste)
    assert gap_fair < gap_base


def test_correlation_remover_kills_linear_leakage():
    X, y, s = _biased_data()
    cr = CorrelationRemover(alpha=1.0).fit(X, sensitive=s)
    Xr = cr.transform(X, sensitive=s)
    # feature 0 was s-dependent; after removal its correlation with s should drop
    before = abs(np.corrcoef(X[:, 0], s)[0, 1])
    after = abs(np.corrcoef(Xr[:, 0], s)[0, 1])
    assert before > 0.3 and after < 0.05


# --- post-processing ------------------------------------------------------

def test_threshold_optimizer_reduces_dp_gap():
    Xtr, ytr, str_, Xte, yte, ste = _split(*_biased_data())
    base = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    opt = ThresholdOptimizer(LogisticRegression(max_iter=500),
                             constraint="demographic_parity").fit(Xtr, ytr, str_)
    gap_base = demographic_parity_difference(base.predict(Xte), ste)
    gap_fair = demographic_parity_difference(opt.predict(Xte, ste), ste)
    assert gap_fair < gap_base


# --- in-processing --------------------------------------------------------

def test_exponentiated_gradient_reduces_dp_gap():
    Xtr, ytr, str_, Xte, yte, ste = _split(*_biased_data())
    base = LogisticRegression(max_iter=500).fit(Xtr, ytr)
    egr = ExponentiatedGradientReduction(LogisticRegression(max_iter=300),
                                         n_iter=15, random_state=0).fit(
        Xtr, ytr, str_)
    gap_base = demographic_parity_difference(base.predict(Xte), ste)
    gap_fair = demographic_parity_difference(egr.predict(Xte), ste)
    assert gap_fair < gap_base
    # and it still classifies better than chance
    assert (egr.predict(Xte) == yte).mean() > 0.6
