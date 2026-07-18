"""H7: conformal v2 (APS, RAPS, jackknife+/CV+, EnbPI) and ensemble uncertainty.

Each set/interval method must achieve (at least) its coverage guarantee on a
held-out set; RAPS must produce smaller sets than APS; the deep ensemble's
disagreement must grow away from the training data (epistemic uncertainty).
"""
import numpy as np
import pytest

from nupyml.inference import APS, RAPS, JackknifePlus, EnbPI, DeepEnsemble
from nupyml.datasets import make_classification, make_regression
from nupyml.linear_model import LogisticRegression, Ridge
from nupyml.ensemble import RandomForestRegressor
from nupyml.model_selection import train_test_split


def test_aps_covers_and_raps_shrinks_sets():
    X, y = make_classification(n_samples=800, n_features=8, n_informative=5,
                               n_classes=4, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0)
    aps = APS(LogisticRegression(max_iter=500), alpha=0.1, random_state=0).fit(Xtr, ytr)
    a_sets = aps.predict_set(Xte)
    a_cov = np.mean([yte[i] in a_sets[i] for i in range(len(yte))])
    a_size = np.mean([len(s) for s in a_sets])
    assert a_cov >= 0.88                              # ~ the 1 - alpha guarantee

    raps = RAPS(LogisticRegression(max_iter=500), alpha=0.1, lam=0.2,
                random_state=0).fit(Xtr, ytr)
    r_sets = raps.predict_set(Xte)
    r_cov = np.mean([yte[i] in r_sets[i] for i in range(len(yte))])
    r_size = np.mean([len(s) for s in r_sets])
    assert r_cov >= 0.85
    assert r_size <= a_size                           # regularisation -> smaller sets


def test_jackknife_plus_meets_cv_plus_coverage():
    X, y = make_regression(n_samples=400, n_features=6, noise=10.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    jk = JackknifePlus(Ridge(alpha=1.0), alpha=0.1, cv=5, random_state=0).fit(Xtr, ytr)
    lo, hi = jk.predict_interval(Xte)
    cov = np.mean((yte >= lo) & (yte <= hi))
    assert cov >= 0.78                                # CV+ guarantees ~1 - 2*alpha
    assert np.all(hi >= lo)


def test_enbpi_covers_with_oob_residuals():
    X, y = make_regression(n_samples=400, n_features=6, noise=10.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    enb = EnbPI(RandomForestRegressor(n_estimators=20, random_state=0), alpha=0.1,
                n_estimators=20, random_state=0).fit(Xtr, ytr)
    lo, hi = enb.predict_interval(Xte)
    assert np.mean((yte >= lo) & (yte <= hi)) >= 0.8


def test_deep_ensemble_uncertainty_grows_off_distribution():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 1)                              # training lives near 0
    y = (2 * X[:, 0] + 0.1 * rng.randn(300))
    de = DeepEnsemble(Ridge(alpha=0.1), n_models=10, random_state=0).fit(X, y)
    near = np.array([[0.0], [0.2], [-0.1]])
    far = np.array([[8.0], [10.0], [-9.0]])            # far outside the training range
    _, std_near = de.predict(near, return_std=True)
    _, std_far = de.predict(far, return_std=True)
    assert std_far.mean() > std_near.mean()           # epistemic uncertainty grows
