"""K2: ensembles v3 -- Mondrian forest, KNORA, super learner, Bayesian model averaging.

The Mondrian forest classifies and learns ONLINE via partial_fit; KNORA selects
competent classifiers per query; the super learner finds convex weights that match
or beat the best base learner; Bayesian model averaging gives posterior weights
that sum to one and favour the better-fitting model.
"""
import numpy as np
import pytest

from sklearn.datasets import make_classification, make_moons, make_friedman1

from nupyml.ensemble import (MondrianForest, KNORA, SuperLearner,
                             BayesianModelAveraging)
from nupyml.linear_model import LinearRegression, Ridge
from nupyml.tree import DecisionTreeClassifier, DecisionTreeRegressor
from nupyml.neighbors import KNeighborsClassifier
from nupyml.svm import SVC
from nupyml.model_selection import train_test_split
from nupyml.metrics import accuracy_score, r2_score


def test_mondrian_forest_batch_and_online():
    X, y = make_classification(n_samples=400, n_features=6, n_informative=4,
                               random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    mf = MondrianForest(n_estimators=25, lifetime=3.0, random_state=0).fit(Xtr, ytr)
    assert accuracy_score(yte, mf.predict(Xte)) > 0.75
    # online: feed the stream in batches via partial_fit
    online = MondrianForest(n_estimators=25, lifetime=3.0, random_state=0)
    for i in range(0, len(Xtr), 50):
        online.partial_fit(Xtr[i:i + 50], ytr[i:i + 50], classes=np.unique(y))
    assert accuracy_score(yte, online.predict(Xte)) > 0.75


@pytest.mark.parametrize("mode", ["union", "eliminate"])
def test_knora_selects_competent_classifiers(mode):
    X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    ests = [DecisionTreeClassifier(max_depth=3, random_state=0),
            KNeighborsClassifier(n_neighbors=5), SVC()]
    kn = KNORA(ests, k=7, mode=mode, random_state=0).fit(Xtr, ytr)
    assert accuracy_score(yte, kn.predict(Xte)) > 0.85


def test_super_learner_matches_or_beats_best_base():
    X, y = make_friedman1(n_samples=400, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    bases = [LinearRegression(), DecisionTreeRegressor(max_depth=5, random_state=0),
             Ridge(alpha=1.0)]
    sl = SuperLearner(bases, random_state=0).fit(Xtr, ytr)
    # weights are a convex combination
    assert abs(sl.weights_.sum() - 1.0) < 1e-6 and np.all(sl.weights_ >= -1e-9)
    best = max(r2_score(yte, b.fit(Xtr, ytr).predict(Xte)) for b in bases)
    assert r2_score(yte, sl.predict(Xte)) >= best - 0.05


def test_bayesian_model_averaging_weights():
    X, y = make_friedman1(n_samples=400, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    bma = BayesianModelAveraging([LinearRegression(), Ridge(alpha=100.0)],
                                 n_params=[11, 11]).fit(Xtr, ytr)
    assert abs(bma.weights_.sum() - 1.0) < 1e-6
    # the better-fitting model (plain OLS here) gets the larger posterior weight
    assert bma.weights_[0] > bma.weights_[1]
    assert r2_score(yte, bma.predict(Xte)) > 0.5
