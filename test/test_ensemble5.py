"""M2: ensembles v5 -- probabilistic RF, Bayesian committee machine,
regression-via-classification, feature-weighted linear stacking.

The probabilistic forest classifies well and widens its uncertainty when the input
noise grows; the committee machine fuses GP experts to a low-error, positive-variance
prediction; regression-via-classification recovers a nonlinear target; feature-
weighted stacking blends base models into a competitive predictor.
"""
import numpy as np

from nupyml.ensemble import (ProbabilisticRandomForest, BayesianCommitteeMachine,
                            RegressionViaClassification,
                            FeatureWeightedLinearStacking)
from nupyml.gaussian_process import GaussianProcessRegressor
from nupyml.tree import DecisionTreeRegressor
from nupyml.linear_model import LinearRegression
from nupyml.datasets import make_classification, make_regression
from nupyml.metrics import r2_score


def test_probabilistic_forest_classifies_and_reflects_input_noise():
    X, y = make_classification(n_samples=250, n_features=6, n_classes=3,
                               n_informative=4, random_state=0)
    Xtr, ytr, Xte = X[:180], y[:180], X[180:]
    calm = ProbabilisticRandomForest(n_estimators=20, feature_sigma=0.1,
                                     random_state=0).fit(Xtr, ytr)
    assert (calm.predict(Xte) == y[180:]).mean() > 0.8
    proba = calm.predict_proba(Xte)
    assert np.allclose(proba.sum(axis=1), 1)
    # more input noise -> less certain votes (higher mean entropy)
    noisy = ProbabilisticRandomForest(n_estimators=20, feature_sigma=1.5,
                                      random_state=0).fit(Xtr, ytr)

    def entropy(p):
        return -(p * np.log(p + 1e-12)).sum(axis=1).mean()

    assert entropy(noisy.predict_proba(Xte)) > entropy(proba)


def test_bayesian_committee_fuses_gp_experts():
    rng = np.random.RandomState(0)
    X = np.linspace(-3, 3, 150).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.1 * rng.randn(150)
    bcm = BayesianCommitteeMachine(
        [GaussianProcessRegressor(alpha=1e-2) for _ in range(3)]).fit(X, y)
    mean, std = bcm.predict(X, return_std=True)
    assert np.mean((mean - np.sin(X).ravel()) ** 2) < 0.01     # accurate fusion
    assert np.all(std > 0)                                     # keeps uncertainty


def test_regression_via_classification_recovers_target():
    X, y = make_regression(n_samples=250, n_features=5, noise=5, random_state=0)
    rvc = RegressionViaClassification(n_bins=12).fit(X[:180], y[:180])
    pred = rvc.predict(X[180:])
    assert r2_score(y[180:], pred) > 0.4
    assert pred.min() >= y.min() - 1 and pred.max() <= y.max() + 1   # decodes in range


def test_feature_weighted_stacking_is_competitive():
    rng = np.random.RandomState(0)
    x = np.sort(rng.uniform(-3, 3, 400)); X = x.reshape(-1, 1)
    y = np.where(x < 0, 2 * x, np.sign(np.sin(3 * x)) * 2) + 0.1 * rng.randn(400)
    tr = rng.rand(400) < 0.6; te = ~tr
    lin = LinearRegression(); tree = DecisionTreeRegressor(max_depth=6)
    fw = FeatureWeightedLinearStacking([lin, tree], alpha=0.1,
                                       random_state=0).fit(X[tr], y[tr])
    lin.fit(X[tr], y[tr]); tree.fit(X[tr], y[tr])
    r_fw = r2_score(y[te], fw.predict(X[te]))
    r_lin = r2_score(y[te], lin.predict(X[te]))
    assert r_fw > r_lin                                        # beats the weaker base
    assert r_fw > 0.8                                          # strong blend
